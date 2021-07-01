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
#include "model.h"
#include "result.h"
#include <functional>
#include <memory>

BEGIN_NS_NNCASE_RUNTIME

struct runtime_tensor_type
{
    const char *data;

    explicit runtime_tensor_type(const char *data) noexcept
        : data(data)
    {
    }

    runtime_tensor_type(runtime_tensor_type &) = delete;
    runtime_tensor_type &operator=(runtime_tensor_type &) = delete;
};

inline bool operator==(runtime_tensor_type &lhs, runtime_tensor_type &rhs) noexcept
{
    return &lhs == &rhs;
}

inline bool operator!=(runtime_tensor_type &lhs, runtime_tensor_type &rhs) noexcept
{
    return &lhs != &rhs;
}

namespace detail
{
class runtime_tensor_impl;
class host_runtime_tensor_impl;
}

class NNCASE_API runtime_tensor
{
public:
    runtime_tensor() noexcept;
    runtime_tensor(std::shared_ptr<detail::runtime_tensor_impl> impl) noexcept;

    datatype_t datatype() const noexcept;
    const runtime_shape_t &shape() const noexcept;
    const runtime_shape_t &strides() const noexcept;
    runtime_tensor_type &tensor_type() const noexcept;
    bool empty() const noexcept;
    bool is_host() const noexcept;
    bool is_contiguous() const noexcept;

    detail::runtime_tensor_impl *impl() noexcept { return impl_.get(); }
    const detail::runtime_tensor_impl *impl() const noexcept { return impl_.get(); }

    bool can_copy_to_without_staging(const runtime_tensor &dest) const noexcept;
    result<void> copy_to(runtime_tensor &dest) noexcept;
    result<runtime_tensor> as_host() noexcept;

    void reset() noexcept;

private:
    std::shared_ptr<detail::runtime_tensor_impl> impl_;
};

NNCASE_API bool operator==(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept;
NNCASE_API bool operator!=(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept;

namespace host_runtime_tensor
{
typedef enum memory_pool_
{
    pool_cpu_only,
    pool_shared
} memory_pool_t;

typedef enum sync_op_
{
    sync_invalidate,
    sync_write_back
} sync_op_t;

typedef enum map_access_
{
    map_none = 0,
    map_read = 1,
    map_write = 2,
    map_read_write = 3
} map_access_t;

DEFINE_ENUM_BITMASK_OPERATORS(map_access_t)

class NNCASE_API mapped_buffer
{
public:
    mapped_buffer() noexcept;
    mapped_buffer(detail::host_runtime_tensor_impl &impl, map_access_t access, uintptr_t address, size_t size_bytes) noexcept;
    mapped_buffer(mapped_buffer &&other) noexcept;
    mapped_buffer(const mapped_buffer &) = delete;
    ~mapped_buffer();

    mapped_buffer &operator=(mapped_buffer &&) noexcept;
    mapped_buffer &operator=(const mapped_buffer &) = delete;

    result<void> unmap() noexcept;

    gsl::span<gsl::byte> buffer() const noexcept
    {
        return { reinterpret_cast<gsl::byte *>(address_), size_bytes_ };
    }

private:
    detail::host_runtime_tensor_impl *impl_;
    map_access_t access_;
    uintptr_t address_;
    size_t size_bytes_;
};

typedef std::function<void(gsl::byte *)> data_deleter_t;

NNCASE_API runtime_tensor_type &tensor_type() noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, bool copy, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, data_deleter_t data_deleter, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, bool copy, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, data_deleter_t data_deleter, memory_pool_t pool = pool_cpu_only, uintptr_t physical_address = 0) noexcept;
NNCASE_API result<memory_pool_t> memory_pool(const runtime_tensor &tensor) noexcept;
NNCASE_API result<mapped_buffer> map(runtime_tensor &tensor, map_access_t access) noexcept;
NNCASE_API result<void> sync(runtime_tensor &tensor, sync_op_t op, bool force = false) noexcept;
}

namespace hrt = host_runtime_tensor;

END_NS_NNCASE_RUNTIME
