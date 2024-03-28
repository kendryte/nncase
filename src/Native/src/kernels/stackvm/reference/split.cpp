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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T>
result<void> split_impl(const T *input, std::span<std::byte *> outputs,
                        std::span<const size_t> in_shape,
                        std::span<const size_t> in_strides,
                        const std::span<strides_t> out_strides, size_t axis,
                        std::span<const size_t> sections,
                        NNCASE_UNUSED kernel_context &context) noexcept {
    for (size_t i = 0; i < outputs.size(); ++i) {
        dims_t out_shape(in_shape);
        out_shape[axis] = sections[i];
        auto output = reinterpret_cast<T *>(outputs[i]);
        size_t sections_sum = 0;
        for (size_t j = 0; j < i; ++j) {
            sections_sum += sections[j];
        }
        try_(kernels::stackvm::apply(
            out_shape, [&](std::span<const size_t> out_index) -> result<void> {
                dims_t in_index(out_index);
                in_index[axis] = sections_sum + out_index[axis];
                output[offset(out_strides[i], out_index)] =
                    input[offset(in_strides, in_index)];
                return ok();
            }));
    }
    return ok();
}
} // namespace

#define SPLIT_IMPL(size, type)                                                 \
    case size:                                                                 \
        return split_impl(reinterpret_cast<const type *>(input), output,       \
                          in_shape, in_strides, out_strides, axis, sections,   \
                          context)

result<void> nncase::kernels::stackvm::reference::split(
    datatype_t type, const std::byte *input, std::span<std::byte *> output,
    std::span<const size_t> in_shape, std::span<const size_t> in_strides,
    std::span<strides_t> out_strides, size_t axis,
    std::span<const size_t> sections, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, SPLIT_IMPL);
}
