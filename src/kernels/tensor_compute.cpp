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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
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

    if (is_contiguous(in_shape, in_strides) && is_contiguous(in_shape, out_strides))
    {
        return cpu::optimized::dequantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
    }
    return cpu::reference::dequantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
}

template result<void> kernels::compare<uint8_t>(compare_op_t op, const uint8_t *input_a, const uint8_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::compare<float>(compare_op_t op, const float *input_a, const float *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::compare<int32_t>(compare_op_t op, const int32_t *input_a, const int32_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::compare<int64_t>(compare_op_t op, const int64_t *input_a, const int64_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> kernels::compare(compare_op_t op, const T *input_a, const T *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::compare(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides);
}

result<void> kernels::lut1d(datatype_t type, const gsl::byte *input, const gsl::byte *table, gsl::byte *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const scalar &min, const scalar &max) noexcept
{
    return cpu::reference::lut1d(type, input, table, output, shape, in_strides, out_strides, min, max);
}

template result<void> kernels::matmul<float>(const float *input_a, const float *input_b, const float *bias, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept;

template <typename T>
result<void> kernels::matmul(const T *input_a, const T *input_b, const T *bias, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept
{
    // if (is_contiguous(in_a_shape, in_a_strides) && is_contiguous(in_b_shape, in_b_strides) && is_contiguous(out_shape, out_strides))
    // {
    return cpu::optimized::matmul(input_a, input_b, bias, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides,
        out_shape, out_strides, fused_activation);
    // }

    return cpu::reference::matmul(input_a, input_b, bias, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides,
        out_shape, out_strides, fused_activation);
}

result<void> kernels::onehot(datatype_t type, const int32_t *indices, gsl::byte *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, gsl::byte *depth, gsl::byte *off_value, gsl::byte *on_value, size_t axis, onehot_mode_t mode, kernel_context &context) noexcept
{
    if (is_contiguous(out_shape, out_strides) && (indices_shape.size() - axis) < 4)
    {
        return cpu::optimized::onehot(type, indices, output, indices_shape, out_shape, out_strides, depth, off_value, on_value, axis, mode, context);
    }
    else
    {
        return cpu::reference::onehot(type, indices, output, indices_shape, out_shape, out_strides, depth, off_value, on_value, axis, mode, context);
    }
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
    if (is_contiguous(in_shape, in_strides) && is_contiguous(in_shape, out_strides))
    {
        return cpu::optimized::quantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
    }
    return cpu::reference::quantize(in_type, out_type, input, output, in_shape, in_strides, out_strides, scale, bias, context);
}

result<void> kernels::transpose(datatype_t type, const gsl::byte *src, gsl::byte *dest, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::transpose(type, src, dest, in_shape, perm, in_strides, out_strides, context);
}

template result<void> kernels::binary<float>(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;
template result<void> kernels::binary<int32_t>(binary_op_t op, const int32_t *input_a, const int32_t *input_b, int32_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;
template result<void> kernels::binary<int64_t>(binary_op_t op, const int64_t *input_a, const int64_t *input_b, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;

template <typename T>
result<void> kernels::binary(binary_op_t op, const T *input_a, const T *input_b, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept
{
    if (is_contiguous(in_a_shape, in_a_strides) && is_contiguous(in_b_shape, in_b_strides) && is_contiguous(out_shape, out_strides))
    {
        // optimization
        if (is_optimized_binary_op(op) && is_optimized_input_shape(in_a_shape, out_shape) && is_optimized_input_shape(in_b_shape, out_shape) && (std::is_same_v<T, float> || std::is_same_v<T, int32_t>))
            return cpu::optimized::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides, fused_activation, context);
    }
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides, fused_activation, context);
}

result<void> kernels::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    if (is_contiguous(shape, in_strides) && is_contiguous(shape, out_strides))
    {
        // optimization
        return cpu::optimized::unary(op, input, output, shape, in_strides, out_strides, context);
    }
    return cpu::reference::unary(op, input, output, shape, in_strides, out_strides, context);
}

template result<void> kernels::reduce<float>(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template result<void> kernels::reduce<int32_t>(reduce_op_t op, int32_t init_value, const int32_t *input, int32_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template result<void> kernels::reduce<int64_t>(reduce_op_t op, int64_t init_value, const int64_t *input, int64_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template <typename T>
result<void> kernels::reduce(reduce_op_t op, T init_value, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    // return cpu::reference::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims, context);
    return cpu::optimized::reduce(op, init_value, input, output, in_shape, axis, in_strides, out_strides, keep_dims, context);
}

template result<void> kernels::reduce_arg<int32_t>(reduce_arg_op_t op, const float *input, int32_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &axis,
    bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template result<void> kernels::reduce_arg<int64_t>(reduce_arg_op_t op, const float *input, int64_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &axis,
    bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template <typename T>
result<void> kernels::reduce_arg(reduce_arg_op_t op, const float *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &axis,
    bool keep_dims, bool select_last_idx, kernel_context &context) noexcept
{
    return cpu::reference::reduce_arg(op, input, output, in_shape, in_strides, out_strides, axis, keep_dims, select_last_idx, context);
}

template result<void> kernels::reduce_prod<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims) noexcept;

template result<void> kernels::reduce_prod<int32_t>(const int32_t *input, int32_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims) noexcept;

template <typename T>
result<void> kernels::reduce_prod(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims) noexcept
{
    return cpu::reference::reduce_prod(input, output, in_shape, in_strides, out_strides, axes, keep_dims);
}

#define DISPATCH_RESIZE(resize_fun)                                                                                                                          \
    runtime_shape_t out_shape { in_shape[0], in_shape[1], static_cast<size_t>(out_h), static_cast<size_t>(out_w) };                                          \
    if (is_contiguous(in_shape, in_strides) && is_contiguous(out_shape, out_strides))                                                                        \
    {                                                                                                                                                        \
        return cpu::optimized::resize_fun(type, input, output, in_shape, in_strides, out_strides, out_h, out_w, align_corners, half_pixel_centers, context); \
    }                                                                                                                                                        \
    else                                                                                                                                                     \
    {                                                                                                                                                        \
        return cpu::reference::resize_fun(type, input, output, in_shape, in_strides, out_strides, out_h, out_w, align_corners, half_pixel_centers, context); \
    }

result<void> kernels::resize_bilinear(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers, kernel_context &context) noexcept
{
    DISPATCH_RESIZE(resize_bilinear);
}

result<void> kernels::resize_nearest_neighbor(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers, kernel_context &context) noexcept
{
    DISPATCH_RESIZE(resize_nearest_neighbor);
}

result<void> kernels::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_axis_t &ends, const runtime_axis_t &strides,
    kernel_context &context) noexcept
{
    bool neg_strides = false;
    for (auto &&stride : strides)
    {
        if (stride < 0)
        {
            neg_strides = true;
            break;
        }
    }
    if (in_strides.size() <= 4 && !neg_strides)
    {
        return cpu::optimized::slice(type, input, output, in_shape, in_strides, out_strides, begins, ends, strides, context);
    }
    else
    {
        return cpu::reference::slice(type, input, output, in_shape, in_strides, out_strides, begins, ends, strides, context);
    }
}

result<void> kernels::gather(datatype_t in_type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis, kernel_context &context) noexcept
{
    if (is_contiguous(in_shape, in_strides) && is_contiguous(out_shape, out_strides))
    {
        return cpu::optimized::gather(in_type, input, output, in_shape, out_shape, in_strides, out_strides, indices, indices_shape, axis, context);
    }
    else
    {
        return cpu::reference::gather(in_type, input, output, in_shape, out_shape, in_strides, out_strides, indices, indices_shape, axis, context);
    }
}

result<void> kernels::gather_nd(datatype_t in_type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t batch_dims, kernel_context &context) noexcept
{
    if (is_contiguous(in_shape, in_strides) && is_contiguous(out_shape, out_strides))
    {
        return cpu::optimized::gather_nd(in_type, input, output, in_shape, out_shape, in_strides, out_strides, indices, indices_shape, batch_dims, context);
    }
    else
    {
        return cpu::reference::gather_nd(in_type, input, output, in_shape, out_shape, in_strides, out_strides, indices, indices_shape, batch_dims, context);
    }
}

template result<void> kernels::cumsum<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    int32_t axis, bool exclusive, bool reverse) noexcept;
template result<void> kernels::cumsum<int32_t>(const int32_t *input, int32_t *output, const runtime_shape_t &in_shape,
    int32_t axis, bool exclusive, bool reverse) noexcept;

template <typename T>
result<void> kernels::cumsum(const T *input, T *output, const runtime_shape_t &in_shape,
    int32_t axis, bool exclusive, bool reverse) noexcept
{
    return cpu::reference::cumsum(input, output, in_shape, axis, exclusive, reverse);
}

template result<void> kernels::hardmax<float>(const float *input, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    float *output, int32_t axis) noexcept;

template <typename T>
result<void> kernels::hardmax(const T *input, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    T *output, int32_t axis) noexcept
{
    return cpu::reference::hardmax(input, in_shape, in_strides, output, axis);
}

template result<void> kernels::random_normal<float>(float *output, const runtime_shape_t &out_shape, float mean, float std, float seed) noexcept;

template <typename T>
result<void> kernels::random_normal(T *output, const runtime_shape_t &out_shape, float mean, float std, float seed) noexcept
{
    return cpu::reference::random_normal(output, out_shape, mean, std, seed);
}

template result<void> kernels::random_uniform<float>(float *output, const runtime_shape_t &out_shape, float low, float high, float seed) noexcept;

template <typename T>
result<void> kernels::random_uniform(T *output, const runtime_shape_t &out_shape, float low, float high, float seed) noexcept
{
    return cpu::reference::random_uniform(output, out_shape, low, high, seed);
}

template result<void> kernels::roi_align<float>(const float *input, const float *rois, int64_t *batch_indices, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, roi_align_mode_t mode, float spatial_scale, int64_t sampling_ratio) noexcept;

template <typename T>
result<void> kernels::roi_align(const T *input, const T *rois, int64_t *batch_indices, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, roi_align_mode_t mode, float spatial_scale, int64_t sampling_ratio) noexcept
{
    return cpu::reference::roi_align(input, rois, batch_indices, output, in_shape, out_shape, mode, spatial_scale, sampling_ratio);
}

template result<void> kernels::sigmoid<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> kernels::sigmoid(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides) noexcept
{
    if (is_contiguous(in_shape, out_strides))
    {
        return cpu::optimized::sigmoid(input, output, in_shape, in_strides, out_strides);
    }

    return cpu::reference::sigmoid(input, output, in_shape, in_strides, out_strides);
}

template result<void> kernels::softmax<float>(const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t axis, float beta) noexcept;

template <typename T>
result<void> kernels::softmax(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t axis, float beta) noexcept
{
    if (is_contiguous(in_shape, in_strides) && is_contiguous(in_shape, out_strides))
    {
        return cpu::optimized::softmax(input, output, in_shape, in_strides, out_strides, axis, beta);
    }
    return cpu::reference::softmax(input, output, in_shape, in_strides, out_strides, axis, beta);
}

template result<void> kernels::ternary<float>(const float *input_a, const float *input_b, const float *input_c, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::ternary<int64_t>(const float *input_a, const int64_t *input_b, const int64_t *input_c, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> kernels::ternary(const float *input_a, const T *input_b, const T *input_c, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept
{
    return cpu::optimized::ternary(input_a, input_b, input_c, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides,
        in_c_shape, in_c_strides, out_strides);
}

template result<void> kernels::topk<float>(const float *input, float *output_values, int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &output_values_shape, const runtime_shape_t &output_values_strides,
    const runtime_shape_t &output_indices_shape, const runtime_shape_t &output_indices_strides,
    const int64_t k, const int32_t axis, const bool largest, const bool sorted) noexcept;

template <typename T>
result<void> kernels::topk(const T *input, T *output_values, int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &output_values_shape, const runtime_shape_t &output_values_strides,
    const runtime_shape_t &output_indices_shape, const runtime_shape_t &output_indices_strides,
    const int64_t k, const int32_t axis, const bool largest, const bool sorted) noexcept
{
    return cpu::reference::topk(input, output_values, output_indices, in_shape, in_strides, output_values_shape, output_values_strides,
        output_indices_shape, output_indices_strides, k, axis, largest, sorted);
}

template result<void> kernels::trilu<float>(const float *input, float *output, const runtime_shape_t &in_shape, const bool upper, const int64_t k) noexcept;

template <typename T>
result<void> kernels::trilu(const T *input, T *output, const runtime_shape_t &in_shape, const bool upper, const int64_t k) noexcept
{
    return cpu::reference::trilu(input, output, in_shape, upper, k);
}

template result<void> kernels::gru<float>(const float *input, const float *w, const float *r, const float *b, float *initial_h, float *output, float *output_h, const runtime_shape_t &input_shape, const runtime_shape_t &w_shape, int mode, bool linear_before_reset) noexcept;

template <typename T>
result<void> kernels::gru(const T *input, const T *w, const T *r, const T *b, T *initial_h, T *output, T *output_h, const runtime_shape_t &input_shape, const runtime_shape_t &w_shape, int mode, bool linear_before_reset) noexcept
{
    return cpu::reference::gru(input, w, r, b, initial_h, output, output_h, input_shape, w_shape, mode, linear_before_reset);
}

template result<void> kernels::tflite_detection_postprocess<float>(const float *boxes, const float *scores, const float *anchors, float *output_locations, float *output_classes, float *output_scores, float *output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale) noexcept;

template <typename T>
result<void> kernels::tflite_detection_postprocess(const T *boxes, const T *scores, const T *anchors, T *output_locations, T *output_classes, T *output_scores, T *output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale) noexcept
{
    return cpu::reference::tflite_detection_postprocess(boxes, scores, anchors, output_locations, output_classes, output_scores, output_num_detections,
        boxes_shape, scores_shape, anchors_shape,
        max_detections, max_classes_per_detection, detections_per_class,
        use_regular_non_max_suppression, nms_score_threshold, nms_iou_threshold,
        num_classes, y_scale, x_scale, h_scale, w_scale);
}

result<void> kernels::space_to_batch(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    return cpu::reference::space_to_batch(type, input, output, in_shape, block_shape, crops, in_strides, out_strides, context);
}

template result<void> kernels::gather_elements(const float *input, const int64_t *indices, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &indices_shape, const int axis) noexcept;

template <typename TI, typename TK>
result<void> kernels::gather_elements(const TI *input, const TK *indices, TI *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &indices_shape, const int axis) noexcept
{
    return cpu::reference::gather_elements(input, indices, output, in_shape, indices_shape, axis);
}

template result<void> kernels::instancenorm<float>(const float *input, float *output, float *scale, float *bias, const runtime_shape_t &in_shape, float epsilon) noexcept;

template <typename T>
result<void> kernels::instancenorm(const T *input, T *output, T *scale, T *bias, const runtime_shape_t &in_shape, float epsilon) noexcept
{
    return cpu::optimized::instancenorm(input, output, scale, bias, in_shape, epsilon);
}

template result<void> kernels::layernorm<float>(const float *input, float *output, float *scale, float *bias, const runtime_shape_t &in_shape, int32_t axis, float epsilon) noexcept;

template <typename T>
result<void> kernels::layernorm(const T *input, T *output, T *scale, T *bias, const runtime_shape_t &in_shape, int32_t axis, float epsilon) noexcept
{
    // return cpu::reference::layernorm(input, output, scale, bias, in_shape, axis, epsilon);
    return cpu::optimized::layernorm(input, output, scale, bias, in_shape, axis, epsilon);
}

template result<void> kernels::compress<float>(const float *input, const uint8_t *condition, float *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis) noexcept;

template <typename T>
result<void> kernels::compress(const T *input, const uint8_t *condition, T *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis) noexcept
{
    return cpu::reference::compress(input, condition, output, input_shape, condition_shape, axis);
}
