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
#include <cstring>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

namespace {
result<buffer_t> allocate_buffer(size_t size_bytes,
                                 hrt::memory_pool_t pool) noexcept {
    buffer_allocate_options options{};

    result<buffer_t> buffer(std::errc::not_enough_memory);
    while (true) {
        options.flags =
            pool == hrt::pool_shared_first || pool == hrt::pool_shared
                ? HOST_BUFFER_ALLOCATE_SHARED
                : HOST_BUFFER_ALLOCATE_CPU_ONLY;
        buffer = buffer_allocator::host().allocate(size_bytes, options);

        if (buffer.is_ok()) {
            return buffer;
        } else if (buffer.is_err()) {
            if (pool == hrt::pool_shared_first) {
                pool = hrt::pool_cpu_only;
            } else {
                return err(buffer.unwrap_err());
            }
        }
    }
}

result<buffer_t> attach_buffer(gsl::span<gsl::byte> data,
                               hrt::data_deleter_t deleter,
                               hrt::memory_pool_t pool) noexcept {
    buffer_attach_options options{};
    options.deleter = std::move(deleter);

    result<buffer_t> buffer(std::errc::not_enough_memory);
    while (true) {
        options.flags =
            pool == hrt::pool_shared_first || pool == hrt::pool_shared
                ? HOST_BUFFER_ATTACH_SHARED
                : 0;
        buffer = buffer_allocator::host().attach(data, options);

        if (buffer.is_ok()) {
            return buffer;
        } else if (buffer.is_err()) {
            if (pool == hrt::pool_shared_first) {
                pool = hrt::pool_cpu_only;
            } else {
                return err(buffer.unwrap_err());
            }
        }
    }
}
} // namespace

result<tensor> runtime::detail::create(datatype_t datatype, dims_t shape,
                                       hrt::memory_pool_t pool) noexcept {
    auto strides = get_default_strides(shape);
    auto size_bytes = compute_size(shape, strides) * get_bytes(datatype);
    checked_try_var(buffer, allocate_buffer(size_bytes, pool));
    return ok(tensor(std::in_place, datatype, std::move(shape),
                     std::move(strides), buffer));
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   strides_t strides,
                                   memory_pool_t pool) noexcept {
    auto size_bytes = compute_size(shape, strides) * get_bytes(datatype);
    checked_try_var(buffer, allocate_buffer(size_bytes, pool));
    return ok(runtime_tensor(tensor(std::in_place, datatype, std::move(shape),
                                    std::move(strides), buffer)));
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   strides_t strides, gsl::span<gsl::byte> data,
                                   bool copy, memory_pool_t pool) noexcept {
    auto size_bytes = compute_size(shape, strides) * get_bytes(datatype);

    // TODO: support strides
    //        std::cout<<size_bytes<<std::endl;
    CHECK_WITH_ERR(data.size_bytes() == size_bytes,
                   std::errc::invalid_argument);

    buffer_t buffer;
    if (copy) {
        checked_try_set(buffer, allocate_buffer(size_bytes, pool));
        checked_try_var(host_buffer, buffer.as<host_buffer_t>());
        checked_try_var(mapped_data, host_buffer->map(map_write));
        auto dest_buffer = mapped_data.buffer();

        // TODO: support strides
        memcpy(dest_buffer.data(), data.data(), size_bytes);
        host_buffer->host_sync_status(host_sync_status_t::need_write_back);
    } else {
        checked_try_set(buffer, attach_buffer(
                                    data, [](gsl::byte *) {}, pool));
    }
    return ok(runtime_tensor(tensor(std::in_place, datatype, std::move(shape),
                                    std::move(strides), buffer)));
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   strides_t strides, gsl::span<gsl::byte> data,
                                   data_deleter_t data_deleter,
                                   memory_pool_t pool) noexcept {
    auto size = compute_size(shape, strides) * get_bytes(datatype);
    CHECK_WITH_ERR(data.size_bytes() == size, std::errc::invalid_argument);

    checked_try_var(buffer, attach_buffer(data, std::move(data_deleter), pool));
    return ok(runtime_tensor(tensor(std::in_place, datatype, std::move(shape),
                                    std::move(strides), buffer)));
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   memory_pool_t pool) noexcept {
    return create(datatype, shape, get_default_strides(shape), pool);
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   gsl::span<gsl::byte> data, bool copy,
                                   memory_pool_t pool) noexcept {
    return create(datatype, shape, get_default_strides(shape), data, copy,
                  pool);
}

result<runtime_tensor> hrt::create(typecode_t datatype, dims_t shape,
                                   gsl::span<gsl::byte> data,
                                   data_deleter_t data_deleter,
                                   memory_pool_t pool) noexcept {
    return create(datatype, shape, get_default_strides(shape), data,
                  std::move(data_deleter), pool);
}

result<hrt::memory_pool_t>
hrt::memory_pool(const runtime_tensor &tensor) noexcept {
    checked_try_var(host_buffer, tensor.impl()->buffer().as_host());
    return ok(host_buffer.has_physical_address() ? pool_shared : pool_cpu_only);
}

result<mapped_buffer> hrt::map(runtime_tensor &tensor,
                               map_access_t access) noexcept {
    checked_try_var(host_buffer, tensor.impl()->buffer().as_host());
    return host_buffer.map(access);
}

result<void> hrt::sync(runtime_tensor &tensor, sync_op_t op,
                       bool force) noexcept {
    checked_try_var(host_buffer, tensor.impl()->buffer().as_host());
    return host_buffer.sync(op, force);
}
