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
#include <nncase/kernels/tensor_compute.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

result<void> kernels::batch_to_space(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::batch_to_space(type, input, output, in_shape, block_shape, crops, in_strides, out_strides);
}

result<void> kernels::broadcast(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::broadcast(type, input, output, in_shape, in_strides, out_shape, out_strides);
}

result<void> kernels::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides) noexcept
{
    return cpu::reference::copy(type, src, dest, shape, src_strides, dest_strides);
}

result<void> kernels::dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias) noexcept
{
    return cpu::reference::dequantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias);
}

result<void> kernels::pad(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, const scalar &pad_value) noexcept
{
    return cpu::reference::pad(type, input, output, in_shape, in_strides, out_strides, paddings, mode, pad_value);
}

result<void> kernels::quantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias) noexcept
{
    return cpu::reference::quantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias);
}

result<void> kernels::transpose(datatype_t type, const gsl::byte *src, gsl::byte *dest, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::transpose(type, src, dest, in_shape, perm, in_strides, out_strides);
}

result<void> kernels::binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_strides, fused_activation);
}

result<void> kernels::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::unary(op, input, output, shape, in_strides, out_strides);
}

result<void> kernels::reduce(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims) noexcept
{
    return cpu::reference::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims);
}

result<void> kernels::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides) noexcept
{
    return cpu::reference::slice(type, input, output, in_shape, in_strides, out_strides, begins, ends, strides);
}
