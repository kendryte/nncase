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
#include "ref_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <nncase/type.h>
#include <nncase/value.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T>
result<void> stack_impl(std::span<const std::byte *const> inputs, T *output,
                        std::span<const size_t> out_shape,
                        std::span<const dims_t> &in_strides,
                        std::span<const size_t> out_strides, size_t axis,
                        NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape,
                 [&](std::span<const size_t> out_index) -> result<void> {
                     auto i = out_index[axis];
                     auto input = IN_CAST(T, inputs[i]);
                     dims_t in_index(out_index);
                     in_index.erase(in_index.begin() + axis);
                     output[offset(out_strides, out_index)] =
                         input[offset(in_strides[i], in_index)];
                     return ok();
                 });
}
} // namespace

#define STACK_IMPL(size, type)                                                 \
    case size:                                                                 \
        return stack_impl(inputs, reinterpret_cast<type *>(output), out_shape, \
                          in_strides, out_strides, axis, context)

result<void> nncase::kernels::stackvm::reference::stack(
    datatype_t type, std::span<const std::byte *const> inputs,
    std::byte *output, std::span<const size_t> out_shape,
    std::span<const dims_t> in_strides, std::span<const size_t> out_strides,
    size_t axis, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, STACK_IMPL);
}