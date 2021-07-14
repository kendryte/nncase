/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "runtime_tensor_impl.h"
#include "shared_runtime_tensor.h"

BEGIN_NS_NNCASE_RUNTIME

namespace detail
{
enum class cache_status_t
{
    valid,
    need_invalidate,
    need_write_back
};

struct host_memory_block
{
    host_runtime_tensor::memory_pool_t pool;
    uintptr_t virtual_address;
    size_t size_bytes;
    host_runtime_tensor::data_deleter_t deleter;
    cache_status_t cache_status;
    physical_memory_block physical_block;

    host_memory_block() = default;
    host_memory_block(const host_memory_block &) = delete;
    host_memory_block(host_memory_block &&other) noexcept;
    host_memory_block &operator=(const host_memory_block &) = delete;
    host_memory_block &operator=(host_memory_block &&other) noexcept;

    ~host_memory_block()
    {
        free();
    }

    void free()
    {
        if (auto d = std::move(deleter))
            d(reinterpret_cast<gsl::byte *>(virtual_address));
        deleter = {};
        physical_block.free(*this);
    }

    gsl::span<gsl::byte> virtual_buffer() const noexcept
    {
        return { reinterpret_cast<gsl::byte *>(virtual_address), size_bytes };
    }
};

class NNCASE_API host_runtime_tensor_impl : public runtime_tensor_impl
{
public:
    host_runtime_tensor_impl(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, host_memory_block memory_block);

    datatype_t datatype() const noexcept override;
    const runtime_shape_t &shape() const noexcept override;
    const runtime_shape_t &strides() const noexcept override;
    runtime_tensor_type &tensor_type() const noexcept override;

    bool can_copy_from_different_type(const runtime_tensor_impl &src) const noexcept override;
    bool can_copy_to_different_type(const runtime_tensor_impl &dest) const noexcept override;

    result<void> copy_to_same_type(runtime_tensor_impl &dest) noexcept override;
    result<void> copy_from_different_type(runtime_tensor_impl &src) noexcept override;
    result<void> copy_to_different_type(runtime_tensor_impl &dest) noexcept override;
    result<void> copy_from_host(runtime_tensor_impl &src) noexcept override;
    result<void> copy_to_host(runtime_tensor_impl &dest) noexcept override;

    result<host_runtime_tensor::mapped_buffer> map(host_runtime_tensor::map_access_t access) noexcept;
    result<void> unmap(host_runtime_tensor::map_access_t access) noexcept;
    result<void> sync(host_runtime_tensor::sync_op_t op, bool force = false) noexcept;
    const host_memory_block &memory_block() const noexcept { return memory_block_; }
    host_memory_block &memory_block() noexcept { return memory_block_; }

private:
    datatype_t datatype_;
    runtime_shape_t shape_;
    runtime_shape_t strides_;
    host_memory_block memory_block_;
};
}

END_NS_NNCASE_RUNTIME
