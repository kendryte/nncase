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
#include <nncase/kernels/stackvm/tensor_ops.h>
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
result<void>
batch_to_space_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                    gsl::span<const size_t> block_shape,
                    const paddings_t &crops, gsl::span<const size_t> in_strides,
                    gsl::span<const size_t> out_strides,
                    NNCASE_UNUSED kernel_context &context) noexcept {
    const auto spatial_dim_start = in_shape.size() - block_shape.size();
    const auto block_size = compute_size(block_shape);
    dims_t batch_reshaped_shape = block_shape;
    batch_reshaped_shape.push_back(in_shape[0] / block_size);
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        // 1. batch reshaped to block_shape[0], ..., block_shape[M-1], batch /
        // prod(block_shape)
        const auto batch_reshaped_index =
            kernels::reshape_linear_index(batch_reshaped_shape, index[0]);
        // 2. permute to [batch / prod(block_shape), input_shape[1],
        // block_shape[0], ..., input_shape[M], block_shape[M-1],
        // input_shape[M+1]]
        // 3. reshaped to [batch / prod(block_shape), input_shape[1] *
        // block_shape[0], ..., input_shape[M] * block_shape[M-1],
        // input_shape[M+1]]
        dims_t spatial_index(block_shape.size());
        for (size_t i = 0; i < spatial_index.size(); i++)
            spatial_index[i] = kernels::linear_index(
                dims_t{in_shape[spatial_dim_start + i], block_shape[i]},
                dims_t{index[spatial_dim_start + i], batch_reshaped_index[i]});

        dims_t out_index = index;
        out_index[0] = batch_reshaped_index.back();
        for (size_t i = 0; i < spatial_index.size(); i++) {
            const auto dim = spatial_index[i];
            const auto spatial_size =
                in_shape[spatial_dim_start + i] * block_shape[i];
            const auto crop_start = (size_t)crops[i].before;
            const auto crop_end = spatial_size - (size_t)crops[i].after;
            if (dim < crop_start || dim >= crop_end)
                return ok();
            out_index[spatial_dim_start + i] = dim - crop_start;
        }

        output[offset(out_strides, out_index)] =
            input[offset(in_strides, index)];
        return ok();
    });
}
} // namespace

#define BATCH_TO_SPACE_IMPL(size, type)                                        \
    case size:                                                                 \
        return batch_to_space_impl(reinterpret_cast<const type *>(input),      \
                                   reinterpret_cast<type *>(output), in_shape, \
                                   block_shape, crops, in_strides,             \
                                   out_strides, context)

result<void>
batch_to_space_impl(datatype_t type, const gsl::byte *input, gsl::byte *output,
                    gsl::span<const size_t> in_shape,
                    gsl::span<const size_t> block_shape,
                    const paddings_t &crops, gsl::span<const size_t> in_strides,
                    gsl::span<const size_t> out_strides,
                    NNCASE_UNUSED kernel_context &context) noexcept {
    switch (runtime::get_bytes(type)) {
        BATCH_TO_SPACE_IMPL(1, uint8_t);
        BATCH_TO_SPACE_IMPL(2, uint16_t);
        BATCH_TO_SPACE_IMPL(4, uint32_t);
        BATCH_TO_SPACE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}

dims_t infer_shape(gsl::span<const size_t> origin_in_shape,
                   gsl::span<const size_t> block_shape,
                   const paddings_t &crops) {
    auto d4 = fixed_dims(0, 2, 3, 1);
    auto d3 = fixed_dims(0, 2, 1);
    auto inPerm = origin_in_shape.size() == 4
                      ? gsl::span<const size_t>{d4.data(), d4.size()}
                      : gsl::span<const size_t>{d3.data(), d3.size()};
    auto in_shape =
        kernels::stackvm::transpose_infer_shape(origin_in_shape, inPerm);
    auto batch = in_shape[0] / compute_size(block_shape);
    auto out_shape = dims_t{batch};
    auto m = block_shape.size();
    for (size_t i = 0; i < m; ++i) {
        auto d = in_shape[i + 1] * block_shape[0] - crops[i].sum();
        out_shape.push_back(d);
    }
    auto remain_size = in_shape.size() - 1 - m;
    if (remain_size > 0) {
        out_shape.insert(out_shape.end(), in_shape.end() - remain_size,
                         in_shape.end());
    }
    auto outd4 = fixed_dims(0, 3, 1, 2);
    auto outd3 = fixed_dims(0, 2, 1);
    auto outPerm = origin_in_shape.size() == 4
                       ? gsl::span<const size_t>{outd4.data(), outd4.size()}
                       : gsl::span<const size_t>{outd3.data(), outd3.size()};
    return kernels::stackvm::transpose_infer_shape(out_shape, outPerm);
}

result<value_t> kernels::stackvm::batch_to_space(value_t input,
                                                 value_t block_shape,
                                                 value_t crops, value_t output,
                                                 kernel_context &context) {
    try_input(input_mem, input);
    try_var(block_shape_value, value_as_dims(block_shape));
    try_var(crops_value, value_as_paddings(crops));
    auto dtype = input_tensor->dtype();
    auto out_shape =
        infer_shape(input_tensor->shape(), block_shape_value, crops_value);
    try_output(out_mem, output, dtype, out_shape);
    try_(batch_to_space_impl(dtype, input_mem, out_mem, input_tensor->shape(),
                             block_shape_value, crops_value,
                             input_tensor->strides(), output_tensor->strides(),
                             context));
    return ok(output);
}
