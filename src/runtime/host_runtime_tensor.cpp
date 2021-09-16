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

host_memory_block::host_memory_block(host_memory_block &&other) noexcept
    : pool(other.pool), virtual_address(other.virtual_address), size_bytes(other.size_bytes), deleter(std::move(other.deleter)), cache_status(other.cache_status), physical_block(std::move(other.physical_block))
{
    other.deleter = {};
}

host_memory_block &host_memory_block::operator=(host_memory_block &&other) noexcept
{
    free();
    pool = other.pool;
    virtual_address = other.virtual_address;
    size_bytes = other.size_bytes;
    deleter = std::move(other.deleter);
    cache_status = other.cache_status;
    physical_block = std::move(other.physical_block);
    other.deleter = {};
    return *this;
}

host_runtime_tensor_impl::host_runtime_tensor_impl(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, host_memory_block memory_block)
    : datatype_(datatype), shape_(std::move(shape)), strides_(std::move(strides)), memory_block_(std::move(memory_block))
{
}

datatype_t host_runtime_tensor_impl::datatype() const noexcept
{
    return datatype_;
}
const runtime_shape_t &host_runtime_tensor_impl::shape() const noexcept
{
    return shape_;
}
const runtime_shape_t &host_runtime_tensor_impl::strides() const noexcept
{
    return strides_;
}
runtime_tensor_type &host_runtime_tensor_impl::tensor_type() const noexcept
{
    return host_runtime_tensor_type_;
}
const quant_param_t &host_runtime_tensor_impl::quant_param() const noexcept
{
    return quant_;
}
void host_runtime_tensor_impl::quant_param(const quant_param_t &quant) noexcept
{
    quant_ = quant;
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
        CHECK_WITH_ERR(status == cache_status_t::valid || status == cache_status_t::need_write_back,
            std::errc::operation_not_permitted);
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

hrt::mapped_buffer::mapped_buffer() noexcept
    : impl_(nullptr), access_(hrt::map_none), address_(0), size_bytes_(0)
{
}

hrt::mapped_buffer::mapped_buffer(host_runtime_tensor_impl &impl, map_access_t access, uintptr_t address, size_t size_bytes) noexcept
    : impl_(&impl), access_(access), address_(address), size_bytes_(size_bytes)
{
}

hrt::mapped_buffer::mapped_buffer(mapped_buffer &&other) noexcept
    : impl_(other.impl_), access_(other.access_), address_(other.address_), size_bytes_(other.size_bytes_)
{
    other.impl_ = nullptr;
}

hrt::mapped_buffer::~mapped_buffer()
{
    unmap().expect("unmap failed");
}

hrt::mapped_buffer &hrt::mapped_buffer::operator=(mapped_buffer &&other) noexcept
{
    unmap().expect("unmap failed");
    impl_ = other.impl_;
    access_ = other.access_;
    address_ = other.address_;
    size_bytes_ = other.size_bytes_;
    other.impl_ = nullptr;
    return *this;
}

result<void> hrt::mapped_buffer::unmap() noexcept
{
    if (impl_)
        return impl_->unmap(access_);
    impl_ = nullptr;
    return ok();
}

runtime_tensor_type &hrt::tensor_type() noexcept
{
    return host_runtime_tensor_type_;
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

result<hrt::memory_pool_t> hrt::memory_pool(const runtime_tensor &tensor) noexcept
{
    CHECK_WITH_ERR(tensor.is_host(), std::errc::invalid_argument);
    return ok(static_cast<const host_runtime_tensor_impl *>(tensor.impl())->memory_block().pool);
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
