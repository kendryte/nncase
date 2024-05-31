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
template <class T, class IndicesT>
result<void> one_hot_impl(const IndicesT *indices, T *output,
                          std::span<const size_t> indices_shape,
                          std::span<const size_t> out_shape,
                          std::span<const size_t> out_strides,
                          NNCASE_UNUSED size_t depth, T off_value, T on_value,
                          size_t axis, runtime::stackvm::one_hot_mode_t mode,
                          NNCASE_UNUSED kernel_context &context) {
    return apply(
        out_shape, [&](std::span<const size_t> out_index) -> result<void> {
            dims_t indices_index(indices_shape.size());
            for (size_t i = 0; i < axis; ++i) {
                indices_index[i] = out_index[i];
            }
            for (size_t i = axis; i < indices_shape.size(); ++i) {
                indices_index[i] = out_index[i + 1];
            }
            auto indices_v = indices[offset(get_default_strides(indices_shape),
                                            indices_index)];
            T out_v;
            auto cur_axis_index = static_cast<int64_t>(out_index[axis]);
            if (indices_v < 0 &&
                mode == runtime::stackvm::one_hot_mode_t::process_neg) {
                out_v = (indices_v + static_cast<int64_t>(out_shape[axis])) ==
                                cur_axis_index
                            ? on_value
                            : off_value;
            } else {
                out_v = indices_v == cur_axis_index ? on_value : off_value;
            }

            output[offset(out_strides, out_index)] = out_v;
            return ok();
        });
}
} // namespace

#define ONEHOT_IMPL(size, type)                                                \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return one_hot_impl(                                               \
                indices_value, reinterpret_cast<type *>(output),               \
                indices_shape, out_shape, out_strides, depth,                  \
                reinterpret_cast<type *>(values)[0],                           \
                reinterpret_cast<type *>(values)[1], axis, mode, context);     \
        });

result<void> nncase::kernels::stackvm::reference::one_hot(
    datatype_t type, datatype_t indices_type, const std::byte *indices,
    std::byte *output, std::span<const size_t> indices_shape,
    std::span<const size_t> out_shape, std::span<const size_t> out_strides,
    size_t depth, std::byte *values, size_t axis,
    runtime::stackvm::one_hot_mode_t mode, kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(type, ONEHOT_IMPL);
}