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
#include "../shape_infer.h"
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
using namespace nncase::kernels::stackvm::reference;
#include <iostream>
namespace {
dims_t get_concat_dims(const std::vector<dims_t> &input_dims, int axis) {
    dims_t concat_dims;
    for (size_t i = 0; i < input_dims.size(); ++i) {
        concat_dims.push_back(input_dims[i][axis]);
    }
    return concat_dims;
}

template <class T>
result<void>
reverse_sequence_impl([[maybe_unused]] datatype_t dt, const T *input, T *output,
                      gsl::span<const size_t> in_shape,
                      gsl::span<const size_t> sequence_lens, int64_t batch_axis,
                      int64_t time_axis, gsl::span<const size_t> in_strides,
                      gsl::span<const size_t> out_strides,
                      NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(in_shape, [&](auto &&out_index) -> result<void> {
        dims_t in_index(out_index);
        auto current_batch = out_index[batch_axis];
        auto reverse_size = sequence_lens[current_batch];
        if (in_index[time_axis] < reverse_size) {
            auto reverse_i = reverse_size - 1 - in_index[time_axis];
            in_index[time_axis] = reverse_i;
        }
        output[offset(out_strides, out_index)] =
            input[offset(in_strides, in_index)];
        return ok();
    });
}
} // namespace

#define REVERSE_SEQUENCE_IMPL(size, type)                                      \
    case size:                                                                 \
        return reverse_sequence_impl(dt, IN_CAST(type, input),                 \
                                     OUT_CAST(type, output), in_shape,         \
                                     sequence_lens, batch_axis, time_axis,     \
                                     in_strides, out_strides, context)

result<void> nncase::kernels::stackvm::reference::reverse_sequence(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> sequence_lens,
    int64_t batch_axis, int64_t time_axis, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    TYPE_IMPL_SELECT(dt, REVERSE_SEQUENCE_IMPL);
}
