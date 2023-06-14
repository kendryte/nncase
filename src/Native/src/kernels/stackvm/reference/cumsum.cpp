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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

namespace {

template <typename T>
result<void> cumsum_impl(const T *input, T *output,
                         gsl::span<const size_t> in_shape, int32_t axis,
                         bool exclusive, bool reverse) noexcept {
    const int32_t rank = in_shape.size();
    assert(rank >= 1);
    assert(axis >= 0);
    assert(axis < rank);

    size_t inner = 1;
    size_t outer = 1;
    size_t depth = 1;
    for (int32_t i = 0; i < rank; i++) {
        if (i < axis)
            inner *= in_shape[i];
        else if (i > axis)
            outer *= in_shape[i];
        else
            depth = in_shape[i];
    }

    for (size_t outer_index = 0; outer_index < outer; outer_index++) {
        size_t outer_index_adj;
        if (reverse)
            outer_index_adj = (outer - 1) - outer_index;
        else
            outer_index_adj = outer_index;

        for (size_t inner_index = 0; inner_index < inner; inner_index++) {
            T accumulator = 0;
            size_t inner_index_adj;
            if (reverse)
                inner_index_adj = (inner - 1) - inner_index;
            else
                inner_index_adj = inner_index;
            for (size_t depth_index = 0; depth_index < depth; depth_index++) {
                size_t depth_index_adj;
                if (reverse)
                    depth_index_adj = (depth - 1) - depth_index;
                else
                    depth_index_adj = depth_index;

                size_t index = outer_index_adj;
                index += inner_index_adj * depth * outer;
                index += depth_index_adj * outer;

                if (exclusive) {
                    output[index] = accumulator;
                    accumulator += input[index];
                } else {
                    accumulator += input[index];
                    output[index] = accumulator;
                }
            }
        }
    }

    return ok();
}

#define CUMSUM_IMPL(_ty)                                                       \
    return cumsum_impl(IN_CAST(_ty, input), OUT_CAST(_ty, output), in_shape,   \
                       axis, exclusive, reverse);

result<void> cumsum_impl(typecode_t typecode, const gsl::byte *input,
                         gsl::byte *output, gsl::span<const size_t> in_shape,
                         int32_t axis, bool exclusive, bool reverse) noexcept {
    TYPE_SELECT(typecode, CUMSUM_IMPL)
}
} // namespace

result<value_t> nncase::kernels::stackvm::cum_sum(
    value_t input, value_t axis, value_t exclusive, value_t reverse,
    value_t output, [[maybe_unused]] kernel_context &context) {
    try_input(input_mem, input);
    try_output(out_mem, output, input_tensor->dtype(), input_tensor->shape());
    try_positive_axis(axis_value, axis, input_tensor);
    try_to_scalar(exclusive_value, exclusive, bool);
    try_to_scalar(reverse_value, reverse, bool);
    try_typecode(typecode, input_tensor);
    try_(cumsum_impl(typecode, input_mem, out_mem, input_tensor->shape(),
                     axis_value, exclusive_value, reverse_value));
    return ok(output);
}
