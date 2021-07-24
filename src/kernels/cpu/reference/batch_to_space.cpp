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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class T>
result<void> batch_to_space_impl(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &block_shape,
    const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto spatial_dim_start = in_shape.size() - block_shape.size();
    const auto block_size = compute_size(block_shape);
    runtime_shape_t batch_reshaped_shape = block_shape;
    batch_reshaped_shape.push_back(in_shape[0] / block_size);
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        // 1. batch reshaped to block_shape[0], ..., block_shape[M-1], batch / prod(block_shape)
        const auto batch_reshaped_index = kernels::reshape_linear_index(batch_reshaped_shape, index[0]);
        // 2. permute to [batch / prod(block_shape), input_shape[1], block_shape[0], ..., input_shape[M], block_shape[M-1], input_shape[M+1]]
        // 3. reshaped to [batch / prod(block_shape), input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1], input_shape[M+1]]
        runtime_shape_t spatial_index(block_shape.size());
        for (size_t i = 0; i < spatial_index.size(); i++)
            spatial_index[i] = kernels::linear_index(
                runtime_shape_t { in_shape[spatial_dim_start + i], block_shape[i] },
                runtime_shape_t { index[spatial_dim_start + i], batch_reshaped_index[i] });

        runtime_shape_t out_index = index;
        out_index[0] = batch_reshaped_index.back();
        for (size_t i = 0; i < spatial_index.size(); i++)
        {
            const auto dim = spatial_index[i];
            const auto spatial_size = in_shape[spatial_dim_start + i] * block_shape[i];
            const auto crop_start = (size_t)crops[i].before;
            const auto crop_end = spatial_size - (size_t)crops[i].after;
            if (dim < crop_start || dim >= crop_end)
                return ok();
            out_index[spatial_dim_start + i] = dim - crop_start;
        }

        output[offset(out_strides, out_index)] = input[offset(in_strides, index)];
        return ok();
    });
}
}

#define BATCH_TO_SPACE_IMPL(size, type) \
    case size:                          \
        return batch_to_space_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, block_shape, crops, in_strides, out_strides, context)

result<void> reference::batch_to_space(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    switch (runtime::get_bytes(type))
    {
        BATCH_TO_SPACE_IMPL(1, uint8_t);
        BATCH_TO_SPACE_IMPL(2, uint16_t);
        BATCH_TO_SPACE_IMPL(4, uint32_t);
        BATCH_TO_SPACE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}
