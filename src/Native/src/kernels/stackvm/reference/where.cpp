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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

template <typename T>
result<void>
where_impl(const bool *cond, const T *x, const T *y, T *output,
           std::span<const size_t> cond_shape, std::span<const size_t> x_shape,
           std::span<const size_t> y_shape, std::span<const size_t> out_shape,
           std::span<const size_t> cond_strides,
           std::span<const size_t> x_strides, std::span<const size_t> y_strides,
           std::span<const size_t> out_strides) {
    return apply(out_shape, [&](const auto &index) -> result<void> {
        const auto cond_index =
            kernels::detail::get_reduced_offset(index, cond_shape);
        const auto x_index =
            kernels::detail::get_reduced_offset(index, x_shape);
        const auto y_index =
            kernels::detail::get_reduced_offset(index, y_shape);

        const auto a = cond[offset(cond_strides, cond_index)];
        const auto b = x[offset(x_strides, x_index)];
        const auto c = y[offset(y_strides, y_index)];

        output[offset(out_strides, index)] = a ? b : c;
        return ok();
    });
}

#define WHERE_IMPL(_ty)                                                        \
    return where_impl(cond, IN_CAST(_ty, x), IN_CAST(_ty, y),                  \
                      OUT_CAST(_ty, output), cond_shape, x_shape, y_shape,     \
                      out_shape, cond_strides, x_strides, y_strides,           \
                      out_strides);

result<void> nncase::kernels::stackvm::reference::where(
    datatype_t dt, const bool *cond, const std::byte *x, const std::byte *y,
    std::byte *output, std::span<const size_t> cond_shape,
    std::span<const size_t> x_shape, std::span<const size_t> y_shape,
    std::span<const size_t> out_shape, std::span<const size_t> cond_strides,
    std::span<const size_t> x_strides, std::span<const size_t> y_strides,
    std::span<const size_t> out_strides) {
    try_var(tycode, to_typecode(dt));
    TYPE_SELECT(tycode, WHERE_IMPL);
}