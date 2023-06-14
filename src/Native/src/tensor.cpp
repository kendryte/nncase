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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>
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
      length_(compute_size(shape_)),
      buffer_(buffer) {
    assert(get_bytes(dtype_, shape_, strides_) == buffer.size_bytes());
}

bool tensor_node::is_contiguous() const noexcept {
    return runtime::is_contiguous(shape_, strides_);
}

result<void> tensor_node::copy_from(tensor src) noexcept {
    return src->copy_to(tensor(this));
}

result<void> tensor_node::copy_to(tensor dest) const noexcept {
    CHECK_WITH_ERR(dtype().equals(dest->dtype()), std::errc::invalid_argument);
    CHECK_WITH_ERR(shape() == dest->shape(), std::errc::invalid_argument);
    return buffer().copy_to(dest->buffer(), dtype(), shape(), strides(),
                            dest->strides());
}

result<tensor> tensor_node::to_host() noexcept {
    if (buffer_.buffer().is_a<host_buffer_t>())
        return ok(tensor(this));
    return err(std::errc::not_supported);
}

result<void> tensor_node::copy_to(value_t dest) const noexcept {
    try_var(dest_tensor, dest.as<tensor>());
    return copy_to(dest_tensor);
}
