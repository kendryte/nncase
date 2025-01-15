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
#include "datatypes.h"
#include "host_buffer.h"
#include "model.h"
#include "result.h"
#include <functional>
#include <memory>
#include <nncase/tensor.h>

BEGIN_NS_NNCASE_RUNTIME

// V1 APIs

class NNCASE_API runtime_tensor {
  public:
    runtime_tensor() noexcept;
    explicit runtime_tensor(tensor impl) noexcept;

    typecode_t datatype() const noexcept;
    gsl::span<const size_t> shape() const noexcept;
    gsl::span<const size_t> strides() const noexcept;
    bool empty() const noexcept;
    bool is_host() const noexcept;
    bool is_contiguous() const noexcept;

    bool can_copy_to_without_staging(const runtime_tensor &dest) const noexcept;
    result<void> copy_to(runtime_tensor &dest) noexcept;
    result<runtime_tensor> to_host() noexcept;

    void reset() noexcept;

    tensor impl() const noexcept { return impl_; }

  private:
    tensor impl_;
};

NNCASE_API bool operator==(const runtime_tensor &lhs,
                           const runtime_tensor &rhs) noexcept;
NNCASE_API bool operator!=(const runtime_tensor &lhs,
                           const runtime_tensor &rhs) noexcept;

namespace host_runtime_tensor {

typedef enum memory_pool_ {
    pool_shared_first,
    pool_cpu_only,
    pool_shared
} memory_pool_t;

typedef std::function<void(gsl::byte *)> data_deleter_t;

NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape,
       memory_pool_t pool = pool_cpu_only) noexcept;
NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape, gsl::span<gsl::byte> data, bool copy,
       memory_pool_t pool = pool_cpu_only,
       uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape, gsl::span<gsl::byte> data,
       data_deleter_t data_deleter, memory_pool_t pool = pool_cpu_only,
       uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape, strides_t strides,
       memory_pool_t pool = pool_cpu_only) noexcept;
NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape, strides_t strides,
       gsl::span<gsl::byte> data, bool copy,
       memory_pool_t pool = pool_cpu_only,
       uintptr_t physical_address = 0) noexcept;
NNCASE_API result<runtime_tensor>
create(typecode_t datatype, dims_t shape, strides_t strides,
       gsl::span<gsl::byte> data, data_deleter_t data_deleter,
       memory_pool_t pool = pool_cpu_only,
       uintptr_t physical_address = 0) noexcept;

NNCASE_API result<memory_pool_t>
memory_pool(const runtime_tensor &tensor) noexcept;
NNCASE_API result<mapped_buffer> map(runtime_tensor &tensor,
                                     map_access_t access) noexcept;
NNCASE_API result<void> sync(runtime_tensor &tensor, sync_op_t op,
                             bool force = false) noexcept;
} // namespace host_runtime_tensor

namespace hrt = host_runtime_tensor;

namespace detail {
NNCASE_API result<tensor>
create(datatype_t datatype, dims_t shape,
       hrt::memory_pool_t pool = hrt::pool_cpu_only) noexcept;
}

END_NS_NNCASE_RUNTIME
