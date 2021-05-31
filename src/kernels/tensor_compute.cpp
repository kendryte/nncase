/* Copyright 2019-2020 Canaan Inc.
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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

result<void> kernels::batch_to_space(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, 
    kernel_context &context) noexcept
{
    return cpu::reference::batch_to_space(type, input, output, in_shape, block_shape, crops, in_strides, out_strides, context);
}

result<void> kernels::broadcast(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::broadcast(type, input, output, in_shape, in_strides, out_shape, out_strides, context);
}

result<void> kernels::concat(datatype_t type, gsl::span<const gsl::byte *const> inputs, gsl::byte *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, 
    kernel_context &context) noexcept
{
    if (in_strides[0].size() <= 4)
    {
        return cpu::optimized::concat(type, inputs, output, out_shape, in_strides, out_strides, axis, concat_dims, context);
    }
    else
    {
        return cpu::reference::concat(type, inputs, output, out_shape, in_strides, out_strides, axis, concat_dims, context);
    }
}

result<void> kernels::convert(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::convert(in_type, out_type, input, output, in_shape, in_strides, out_strides, context);
}

result<void> kernels::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides, kernel_context &context) noexcept
{
    const auto default_strides = get_default_strides(shape);
    auto src_dims_offset = get_last_not_contiguous_index(src_strides, default_strides);
    auto dest_dims_offset = get_last_not_contiguous_index(dest_strides, default_strides);

    auto src_continuous = src_strides == default_strides;
    auto dest_continuous = dest_strides == default_strides;
    if (!src_continuous && !dest_continuous)
    {
        return cpu::reference::copy(type, src, dest, shape, src_strides, dest_strides, context);
    }
    else
    {
        copy_impl_select select = copy_impl_select::all_contiguous;
        int dims_offset = -1;
        if (src_continuous && dest_continuous)
        {
            select = copy_impl_select::all_contiguous;
            dims_offset = -1;
        }
        else if (src_continuous && dest_dims_offset < 5)
        {
            select = copy_impl_select::src_contiguous;
            dims_offset = dest_dims_offset;
        }
        else if (dest_continuous && src_dims_offset < 5)
        {
            select = copy_impl_select::dest_contiguous;
            dims_offset = src_dims_offset;
        }
        return cpu::optimized::copy(type, src, dest, shape, src_strides, dest_strides, dims_offset, select, context);
    }
}

result<void> kernels::dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context) noexcept
{
    return cpu::reference::dequantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
}

result<void> kernels::lut1d(datatype_t type, const gsl::byte *input, const gsl::byte *table, gsl::byte *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const scalar &min, const scalar &max) noexcept
{
    return cpu::reference::lut1d(type, input, table, output, shape, in_strides, out_strides, min, max);
}

result<void> kernels::pad(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, 
    const scalar &pad_value, kernel_context &context) noexcept
{
    return cpu::reference::pad(type, input, output, in_shape, in_strides, out_strides, paddings, mode, pad_value, context);
}

result<void> kernels::quantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context) noexcept
{
    return cpu::reference::quantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
}

result<void> kernels::transpose(datatype_t type, const gsl::byte *src, gsl::byte *dest, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::transpose(type, src, dest, in_shape, perm, in_strides, out_strides, context);
}

result<void> kernels::binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation, 
    kernel_context &context) noexcept
{
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_strides, fused_activation, context);
}

result<void> kernels::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::unary(op, input, output, shape, in_strides, out_strides, context);
}

result<void> kernels::reduce(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    return cpu::reference::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims, context);
}

result<void> kernels::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
    kernel_context &context) noexcept
{
    if (in_strides.size() <= 4)
    {
        return cpu::optimized::slice(type, input, output, in_shape, in_strides, out_strides, begins, ends, strides, context);
    }
    else
    {
        return cpu::optimized::slice(type, input, output, in_shape, in_strides, out_strides, begins, ends, strides, context);
    }
}
