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
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/tensor.h>
#include <numeric>

using namespace nncase;
using namespace nncase::runtime;

tensor_node::tensor_node(datatype_t dtype, dims_t shape, strides_t strides,
                         runtime::buffer_slice buffer)
    : dtype_(std::move(dtype)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      length_(std::reduce(shape_.begin(), shape_.end(), size_t(1),
                          std::multiplies<>())),
      buffer_(buffer) {
    assert(dtype_->size_bytes() * shape_.front() * strides_.front() ==
           buffer.size_bytes());
}

bool tensor_node::is_contiguous() const noexcept {
    return strides() == get_default_strides(shape());
}

result<void> tensor_node::copy_from(tensor src) const noexcept {
    return err(std::errc::not_supported);
}

result<void> tensor_node::copy_to(tensor dest) const noexcept {
    return err(std::errc::not_supported);
}

result<tensor> tensor_node::to_host() const noexcept {
    return err(std::errc::not_supported);
}
