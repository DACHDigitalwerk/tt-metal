// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {

class GlobalSemaphore;
namespace distributed {
class MeshCommandQueue;
class MeshDevice;
}  // namespace distributed
struct GlobalSemaphoreSlotLease;

// GlobalSemaphoreImpl is implemented as a wrapper around a sharded buffer, or around a slot of the
// MeshDevice's GlobalSemaphore pool.
// This can be updated in the future to be its own container with optimized dispatch functions
class GlobalSemaphoreImpl {
public:
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint8_t cq_id);

    // Dedicated constructor for creating a global semaphore **without allocation**.
    // The instantiation of GlobalSemphore will be emplaced onto the address specified.
    GlobalSemaphoreImpl(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        uint64_t address,
        uint8_t cq_id);

    // Takes a slot from the pool of `cq`'s device and sets it to 0 through `cq`.
    GlobalSemaphoreImpl(distributed::MeshCommandQueue& cq, CoreRangeSet cores);

    // Copy/move semantics
    GlobalSemaphoreImpl(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl& operator=(const GlobalSemaphoreImpl&) = default;
    GlobalSemaphoreImpl(GlobalSemaphoreImpl&&) noexcept = default;
    GlobalSemaphoreImpl& operator=(GlobalSemaphoreImpl&&) noexcept = default;

    // Throws if the MeshDevice has been destroyed.
    distributed::MeshDevice& device() const;

    const CoreRangeSet& cores() const;

    BufferType buffer_type() const;

    DeviceAddr address() const;

    bool is_pool_backed() const { return lease_ != nullptr; }

    // `cq_id` is only used (and only looked up) with fast dispatch; it is ignored on slow dispatch / simulator.
    void reset_semaphore_value(uint32_t reset_value, uint8_t cq_id) const;

private:
    void setup_buffer(
        distributed::MeshDevice& device,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        std::optional<uint64_t> address,
        uint8_t cq_id);
    std::shared_ptr<distributed::MeshDevice> lock_device() const;

    std::shared_ptr<distributed::MeshBuffer> buffer_;
    std::shared_ptr<const GlobalSemaphoreSlotLease> lease_;
    std::weak_ptr<distributed::MeshDevice> device_;
    CoreRangeSet cores_;
};

}  // namespace tt::tt_metal
