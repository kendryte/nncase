/* Copyright 2020 Canaan Inc.
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
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

namespace
{
runtime_tensor_type host_runtime_tensor_type_ { "host" };
}

bool host_runtime_tensor_impl::can_copy_from_different_type(NNCASE_UNUSED const runtime_tensor_impl &src) const noexcept
{
    return true;
}

bool host_runtime_tensor_impl::can_copy_to_different_type(NNCASE_UNUSED const runtime_tensor_impl &dest) const noexcept
{
    return true;
}

result<void> host_runtime_tensor_impl::copy_to_same_type(NNCASE_UNUSED runtime_tensor_impl &dest) noexcept
{
    CHECK_WITH_ERR(datatype() == dest.datatype(), nncase_errc::datatype_mismatch);
    CHECK_WITH_ERR(shape() == dest.shape(), nncase_errc::shape_mismatch);

    try_var(src_map, map(hrt::map_read));
    try_var(dest_map, static_cast<host_runtime_tensor_impl &>(dest).map(hrt::map_write));
    return kernels::copy(datatype(), src_map.buffer().data(), dest_map.buffer().data(), shape(), strides(), dest.strides());
}

result<void> host_runtime_tensor_impl::copy_from_different_type(NNCASE_UNUSED runtime_tensor_impl &src) noexcept
{
    return src.copy_to_host(*this);
}

result<void> host_runtime_tensor_impl::copy_to_different_type(NNCASE_UNUSED runtime_tensor_impl &dest) noexcept
{
    return dest.copy_from_host(*this);
}

result<void> host_runtime_tensor_impl::copy_from_host(NNCASE_UNUSED runtime_tensor_impl &src) noexcept
{
    return src.copy_to_same_type(*this);
}

result<void> host_runtime_tensor_impl::copy_to_host(NNCASE_UNUSED runtime_tensor_impl &dest) noexcept
{
    return copy_to_same_type(dest);
}

result<hrt::mapped_buffer> host_runtime_tensor_impl::map(hrt::map_access_t access) noexcept
{
    auto status = memory_block_.cache_status;
    if (status == cache_status_t::need_invalidate)
        try_(sync(hrt::sync_invalidate));
    return ok(hrt::mapped_buffer(*this, access, memory_block_.virtual_address, memory_block_.size_bytes));
}

result<void> host_runtime_tensor_impl::unmap(hrt::map_access_t access) noexcept
{
    if (access & hrt::map_write)
    {
        auto status = memory_block_.cache_status;
        assert(status == cache_status_t::valid || status == cache_status_t::need_write_back);
        memory_block_.cache_status = cache_status_t::need_write_back;
    }

    return ok();
}

result<void> host_runtime_tensor_impl::sync(hrt::sync_op_t op, bool force) noexcept
{
    if (memory_block_.pool == hrt::pool_cpu_only)
        return ok();

    bool do_sync = false;
    if (force)
    {
        do_sync = true;
    }
    else
    {
        auto status = memory_block_.cache_status;
        if (op == hrt::sync_write_back)
        {
            CHECK_WITH_ERR(status == cache_status_t::valid || status == cache_status_t::need_write_back,
                std::errc::operation_not_permitted);
            if (status == cache_status_t::need_write_back)
                do_sync = true;
        }
        else
        {
            CHECK_WITH_ERR(status == cache_status_t::valid || status == cache_status_t::need_invalidate,
                std::errc::operation_not_permitted);
            if (status == cache_status_t::need_invalidate)
                do_sync = true;
        }
    }

    if (do_sync)
    {
        try_(physical_memory_block::sync(memory_block_, op));
        memory_block_.cache_status = cache_status_t::valid;
    }

    return ok();
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    host_memory_block block {};
    block.pool = pool;
    block.size_bytes = compute_size(shape, strides) * get_bytes(datatype);

    if (pool == pool_cpu_only)
    {
        std::unique_ptr<gsl::byte[]> buffer(new (std::nothrow) gsl::byte[block.size_bytes]);
        CHECK_WITH_ERR(buffer, std::errc::not_enough_memory);
        block.deleter = std::default_delete<gsl::byte[]>();
        block.virtual_address = (uintptr_t)buffer.release();
    }
    else
    {
        if (physical_address)
        {
            block.physical_block.physical_address = physical_address;
            try_(physical_memory_block::acknowledge(block));
        }
        else
        {
            try_(physical_memory_block::allocate(block));
        }
    }

    std::shared_ptr<runtime_tensor_impl> impl(new (std::nothrow) host_runtime_tensor_impl(datatype,
        std::move(shape), std::move(strides), std::move(block)));
    CHECK_WITH_ERR(impl, std::errc::not_enough_memory);
    return ok(runtime_tensor(std::move(impl)));
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, bool copy, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    auto size = compute_size(shape, strides) * get_bytes(datatype);
    CHECK_WITH_ERR(data.size_bytes() == size, std::errc::invalid_argument);

    host_memory_block block {};
    block.pool = pool;
    block.size_bytes = size;

    if (pool == pool_cpu_only)
    {
        if (copy)
        {
            std::unique_ptr<gsl::byte[]> buffer(new (std::nothrow) gsl::byte[block.size_bytes]);
            CHECK_WITH_ERR(buffer, std::errc::not_enough_memory);
            try_(kernels::copy(datatype, data.data(), buffer.get(), shape, strides, strides));
            block.deleter = std::default_delete<gsl::byte[]>();
            block.virtual_address = (uintptr_t)buffer.release();
        }
        else
        {
            block.virtual_address = (uintptr_t)data.data();
        }
    }
    else
    {
        if (!copy)
            block.virtual_address = (uintptr_t)data.data();
        block.physical_block.physical_address = physical_address;
        if (block.virtual_address || block.physical_block.physical_address)
        {
            try_(physical_memory_block::acknowledge(block));
        }
        else
        {
            try_(physical_memory_block::allocate(block));
        }

        if (copy)
        {
            CHECK_WITH_ERR(block.virtual_address, std::errc::not_enough_memory);
            try_(kernels::copy(datatype, data.data(), block.virtual_buffer().data(), shape, strides, strides));
            block.cache_status = cache_status_t::need_write_back;
        }
    }

    std::shared_ptr<runtime_tensor_impl> impl(new (std::nothrow) host_runtime_tensor_impl(datatype,
        std::move(shape), std::move(strides), std::move(block)));
    CHECK_WITH_ERR(impl, std::errc::not_enough_memory);
    return ok(runtime_tensor(std::move(impl)));
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, data_deleter_t data_deleter, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    auto size = compute_size(shape, strides) * get_bytes(datatype);
    CHECK_WITH_ERR(data.size_bytes() == size, std::errc::invalid_argument);

    host_memory_block block {};
    block.pool = pool;
    block.size_bytes = size;
    block.deleter = std::move(data_deleter);

    if (pool == pool_cpu_only)
    {
        block.virtual_address = (uintptr_t)data.data();
    }
    else
    {
        block.virtual_address = (uintptr_t)data.data();
        block.physical_block.physical_address = physical_address;
        try_(physical_memory_block::acknowledge(block));
    }

    std::shared_ptr<runtime_tensor_impl> impl(new (std::nothrow) host_runtime_tensor_impl(datatype,
        std::move(shape), std::move(strides), std::move(block)));
    CHECK_WITH_ERR(impl, std::errc::not_enough_memory);
    return ok(runtime_tensor(std::move(impl)));
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    return create(datatype, shape, get_default_strides(shape), pool, physical_address);
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, bool copy, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    return create(datatype, shape, get_default_strides(shape), data, copy, pool, physical_address);
}

result<runtime_tensor> hrt::create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, data_deleter_t data_deleter, memory_pool_t pool, uintptr_t physical_address) noexcept
{
    return create(datatype, shape, get_default_strides(shape), data, std::move(data_deleter), pool, physical_address);
}

result<hrt::mapped_buffer> hrt::map(runtime_tensor &tensor, map_access_t access) noexcept
{
    CHECK_WITH_ERR(tensor.is_host(), std::errc::invalid_argument);
    return static_cast<host_runtime_tensor_impl *>(tensor.impl())->map(access);
}

result<void> hrt::sync(runtime_tensor &tensor, sync_op_t op, bool force) noexcept
{
    CHECK_WITH_ERR(tensor.is_host(), std::errc::invalid_argument);
    return static_cast<host_runtime_tensor_impl *>(tensor.impl())->sync(op, force);
}
