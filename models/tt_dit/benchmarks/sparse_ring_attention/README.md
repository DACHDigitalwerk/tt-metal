# Sparse ring attention benchmark (Blackhole Galaxy, SGLang-compatible inputs)

Per-call attention timing and accuracy of `ttnn.transformer.sparse_ring_sdpa` (block-sparse ring attention) on a
4x8 Blackhole Galaxy, on the same Q / K / V tensors an SGLang attention backend receives. The workload is
MiniMax-H3's DiT joint attention (text, condition keyframes, audio and video packed into one non-causal sequence;
56 heads, head dim 128). Results go into a JSONL schema that any other platform can fill in for a side-by-side
table (`merge_results.py`).

## What is measured

One attention call: Q / K / V in, attention output out. Projections, norms, RoPE and the rest of the layer are not
included. Every backend runs on the same inputs and shapes:

| length | tokens (this geometry) | rows per device (SP=8) | heads | head dim |
|---|---|---|---|---|
| 5 s | 37,888 | 4,736 | 56 (14 per device, TP=4) | 128 |
| 10 s | 73,472 | 9,184 | 56 | 128 |
| 15 s | 109,312 | 13,664 | 56 | 128 |

The sequence is sharded over the 8 devices of the ring axis, heads over the 4 devices of the other axis. Captured
inputs with fewer tokens (e.g. 37,760 at 5 s) are padded to the next working point. The padding is masked
(`logical_n`), so it is not attended.

Timing: one eager call (compiles and is used for the accuracy check), one trace capture, then `--warmup` + `--reps`
trace replays. Each replay is timed on the host from `execute_trace` to `synchronize_device`, so dispatch is
excluded and the time is device time plus one synchronize. We report the median, p10 and p90 over the replays.

Accuracy: every backend's output is compared with exact attention computed in fp64 on the host from the same bf16
inputs. The comparison uses a slice (default: first and last head, first and last SP shard; shard 0 holds the
text prefix), over real tokens only:

- `rel_l2_vs_dense` = ||out - ref|| / ||ref||
- `pcc_vs_dense` = Pearson correlation of out and ref
- `scale_vs_dense` = <out, ref> / <ref, ref> (fitted output scale; PCC does not catch a scale error)

For the sparse backends, `extra.pcc_vs_block_sparse_ref` also checks the op against an fp64 reference of the
block-sparse result it is meant to compute, using the same mask. This is a kernel-correctness check, separate from
the sparsity error. `extra.max_abs_diff_traced_vs_eager` should be 0.

## Backends

`--attention-backend` takes SGLang's backend names (a comma-separated list runs several on each input):

| backend | what runs on the Galaxy | timed call |
|---|---|---|
| `fa` | dense ring attention (`ring_joint_scaled_dot_product_attention`), bf16, HiFi2 | the op |
| `sol_attn` | Sol pre-pass on device (block means, pooled K / V all-gathers, diag-threshold routing to a 64x64 block mask), then `sparse_ring_sdpa` pooled route: kept blocks exact, every other block one pooled entry (mean K, mean V, weighted by its token count) | pre-pass + op |
| `video_sparse_attn` | `sparse_ring_sdpa` kept-only (VSA fine branch) on a block mask from a **host placeholder** scorer (see below) | the op only |

`--attention-backend-config` uses SGLang's syntax (JSON or `k=v,k=v`) and applies to the sparse backends:

- `sol_attn`: `tau` (1.0), `thresh_type` (`diag`, the only one supported), `sink_start` (0, the only value supported:
  the prefix is packed first), `sink_tokens` (0; an int, or `"text_len"` to use the input's text length). These are
  exact key-sink columns. `kv_splits` is recorded but has no effect (GPU kernel tuning). Options specific to this
  implementation: `band` (1: the Q block's own and adjacent key blocks are always exact), `dense_prefix_queries`
  (false: when true, query blocks inside the sink prefix attend densely through the mask), `k_chunk` (256).
- `video_sparse_attn`: `VSA_sparsity` (0.9), `vsa_mode` (`exempt` | `compete`), `k_chunk` (256).
- `dense_steps`, `dense_layers`, `dense_backend` are pipeline settings: they choose which calls go to the dense
  backend. They are recorded and listed in `extra.pipeline_keys_not_applied`. To estimate a full generation, weight
  the per-call times by the share of sparse calls.

Flags: `--lofi` (LoFi matmuls in the sparse op; the dense baseline stays HiFi2), `--kv-dtype bf8` (the sparse op
reads bfloat8_b K / V and pooled K / V; the pre-pass still reads bf16).

Differences from the GPU `sol_attn` kernel that may affect accuracy or kept fraction: 64-token blocks within each
SP shard, so the last block of a shard can be partial (partial blocks are always pooled, never exact); block 0 of
every shard is always exact; bf16 proxy scores.

### VSA (in development)

> **PLACEHOLDER.** On-device VSA scoring is still being written. Until it lands, `vsa_block_mask_host()` in
> `bench_tt.py` builds the block mask on the host: it mean-pools Q and K per 64-token block, scores q_bar . k_bar
> per head, and keeps the top `ceil((1 - VSA_sparsity) * candidates)` key blocks for each query block. Query blocks
> that touch the non-video prefix attend densely. In `exempt` mode every non-video key block is always kept and
> the candidates are the video key blocks; in `compete` mode all key blocks are candidates. It uses contiguous
> 64-token blocks, not VSA's 4x4x4 cubes, and has no coarse branch or gating. Its time is reported separately as
> `extra.scoring_ms` (host, not comparable). The headline `ms_*` is the attention call alone. Replace that one
> function when the device scorer is available.

The non-video prefix length comes from the input's `segments` metadata (everything before `video`), else
`dense_tokens`, else `--dense-tokens`.

## Input format

One safetensors file per attention call (`--inputs` takes files and/or directories of `*.safetensors`):

| key | shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `[B, T, H, D]` (SGLang NHD / BTHD), H = 56, D = 128, the full sequence and all heads | bf16 |

Optional string metadata: `layer`, `step`, `seconds`, `text_len`, `segments` (JSON object of segment lengths in
packed order, e.g. `{"text": 512, "condition": 0, "audio": 760, "video": 36488}`), `dense_tokens`. `--batch-index`
picks the batch entry (CFG-batched calls have B = 2).

`--synthetic --seconds N --seed S` generates seeded `randn` bf16 tensors at the working point (every row real,
`dense_tokens` = 453 / 849 / 1245). `--save-inputs file.safetensors` writes them in this format, so another
platform can run on exactly the same tensors. Random data routes differently from real activations, so use
captured tensors for accuracy numbers.

### Capturing inputs from SGLang

`capture_qkv_sglang.py` wraps any `sglang` command. It hooks the `forward(query, key, value, attn_metadata)` of
the attention backend impls under `sglang.multimodal_gen.runtime.layers.attention.backends` through a generated
`sitecustomize.py`, so worker processes are hooked too, and writes the selected calls in the format above:

```bash
python models/tt_dit/benchmarks/sparse_ring_attention/capture_qkv_sglang.py \
    --out /data/h3_qkv --layers 2,20,40 --steps 12,30 --seconds 5 \
    --segments text=<n>,condition=<n>,audio=<n>,video=<n> -- \
    sglang generate --model-path <MiniMax-H3 checkpoint> --prompt "<prompt>" --attention-backend fa <5 s T2VA args>
```

Assumptions are listed in the script's docstring. The main ones: run on one GPU (no sequence or tensor
parallelism), so a call sees the full sequence; layer and step are counted per backend-impl instance; calls are
filtered by 56 heads x 128 and at least 4,096 tokens.

## Result schema (JSONL, one record per input x backend)

| field | meaning |
|---|---|
| `platform` | `tenstorrent`, or e.g. `nvidia` |
| `device` | e.g. `Blackhole Galaxy 4x8 (TP=4 heads, SP=8 sequence)`, `H100 SXM 80GB` |
| `backend` | SGLang backend name: `fa`, `sol_attn`, `video_sparse_attn`, ... |
| `config` | the backend config dict as applied (defaults filled in) |
| `seconds` | 5 / 10 / 15, or null |
| `tokens` | real tokens T |
| `heads`, `head_dim` | 56, 128 |
| `dtype` | e.g. `bf16`, `q bf16, kv bf8 lofi` |
| `ms_median`, `ms_p10`, `ms_p90` | per-call attention time over the timed reps, ms |
| `kept_fraction` | kept share of the keepable key blocks, averaged over query blocks and heads (1.0 for dense) |
| `rel_l2_vs_dense`, `pcc_vs_dense`, `scale_vs_dense` | vs exact dense attention on the same inputs (see above) |
| `commit` | source commit (`+dirty` if modified) |
| `timestamp` | UTC ISO 8601 |

Optional fields: `input`, `layer`, `step`, `reps`, `extra` (platform-specific details). A GPU record from FA /
`sol_attn` on the same safetensors files, timed with CUDA events over warm calls, fits this schema directly. If
the GPU reference is the FA bf16 output instead of exact attention, say so in `extra`; FA's own error vs exact is
at the 1e-3 relative-L2 level.

## Reproducing on a Galaxy

tt-metal: this branch (`sosborne/sparse-ring-attn-bench`) with the Sol pre-pass branch merged in. The pre-pass
(`models/tt_dit/models/transformers/minimax_h3/sol_prepass.py` and `sol_kernels/`) comes from
`losullivan/sol-attn-proxy-score` at `214db3f17619`, which merges without conflicts. Once that branch is in
`main`, the merge is not needed.

```bash
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git checkout sosborne/sparse-ring-attn-bench
git merge --no-edit 214db3f17619   # Sol pre-pass, until it is on main
git submodule update --init --recursive
./build_metal.sh && ./create_venv.sh && source python_env/bin/activate
pip install safetensors
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
```

Run (the bench opens the 4x8 mesh with a 1D ring fabric itself):

```bash
B=models/tt_dit/benchmarks/sparse_ring_attention
# synthetic, all three lengths, dense + Sol + VSA placeholder; also save the inputs for other platforms
for s in 5 10 15; do
  python $B/bench_tt.py --synthetic --seconds $s --save-inputs /data/syn_${s}s.safetensors \
      --attention-backend fa,sol_attn,video_sparse_attn --output results_tt.jsonl
done
# captured tensors; sink = text length, as SGLang's sol_attn with sink_tokens=<text length>
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend fa,sol_attn --output results_tt.jsonl
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend sol_attn \
    --attention-backend-config '{"tau": 1.0, "sink_tokens": "text_len"}' --output results_tt.jsonl
# LoFi with bf8 K / V
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend sol_attn --lofi --kv-dtype bf8 --output results_tt.jsonl
# VSA placeholder at the base sparsity, both modes
for mode in exempt compete; do
  python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend video_sparse_attn \
      --attention-backend-config VSA_sparsity=0.9,vsa_mode=$mode --output results_tt.jsonl
done

python $B/merge_results.py results_tt.jsonl results_gpu.jsonl --latest
```

Expected output: one log line per input and backend, for example

```
syn_5s... sol_attn: <ms> ms (p10 <ms>, p90 <ms>), kept 0.xxx, rel L2 0.0xxx, pcc 0.99xxx, scale 0.99xx; {...
  'pcc_vs_block_sparse_ref': 0.9999x, 'prepass_ms_median': ..., 'attention_only_ms_median': ...,
  'max_abs_diff_traced_vs_eager': 0.0}
```

`merge_results.py` then prints one markdown table per input, with a `vs fa` speedup column against the dense row
of the same platform. Each record takes up to a few minutes at 15 s, mostly for the host fp64 reference;
`--ref-heads` / `--ref-shards` shrink the slice. Compare timings only from the same machine and run: Galaxy
power limits differ between systems.

### Full matrix

`run_matrix.sh` runs every case worth comparing, at 5, 10 and 15 s (11 runs per length, appended to
`results_tt.jsonl`, then one table):

| Backend | Settings | dtypes |
|---|---|---|
| `fa` | dense baseline | bf16 |
| `sol_attn` | `tau` 1.0, 1.5, 2.0 (higher keeps fewer blocks) | bf16, and LoFi with bf8 K/V |
| `sol_attn` | `tau` 1.0 with `dense_prefix_queries` (prefix rows dense, as VSA) | bf16 |
| `video_sparse_attn` | `VSA_sparsity` 0.9 (base H3) and 0.8 (distilled) | bf16; 0.9 also LoFi with bf8 K/V |

```bash
./$B/run_matrix.sh                       # all lengths, synthetic inputs (about an hour; 15 s dominates)
LENGTHS=5 ./$B/run_matrix.sh             # one length
INPUTS=/data/h3_qkv ./$B/run_matrix.sh   # captured tensors
```

On synthetic inputs Sol's kept fraction at a given `tau` does not reflect real activations, so the `tau` sweep there
gives time against kept fraction; accuracy needs captured inputs.
