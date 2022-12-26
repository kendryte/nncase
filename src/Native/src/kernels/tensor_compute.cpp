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

result<void> kernels::batch_to_space(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &block_shape,
    const runtime_paddings_t &crops, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, kernel_context &context) noexcept {
    return cpu::reference::batch_to_space(type, input, output, in_shape,
                                          block_shape, crops, in_strides,
                                          out_strides, context);
}

result<void> kernels::broadcast(datatype_t type, const gsl::byte *input,
                                gsl::byte *output,
                                const runtime_shape_t &in_shape,
                                const runtime_shape_t &in_strides,
                                const runtime_shape_t &out_shape,
                                const runtime_shape_t &out_strides,
                                kernel_context &context) noexcept {
    return cpu::reference::broadcast(type, input, output, in_shape, in_strides,
                                     out_shape, out_strides, context);
}

result<void> kernels::concat(datatype_t type,
                             gsl::span<const gsl::byte *const> inputs,
                             gsl::byte *output,
                             const runtime_shape_t &out_shape,
                             gsl::span<const runtime_shape_t> in_strides,
                             const runtime_shape_t &out_strides, size_t axis,
                             const runtime_shape_t &concat_dims,
                             kernel_context &context) noexcept {
    if (in_strides[0].size() <= 4) {
        return cpu::optimized::concat(type, inputs, output, out_shape,
                                      in_strides, out_strides, axis,
                                      concat_dims, context);
    } else {
        return cpu::reference::concat(type, inputs, output, out_shape,
                                      in_strides, out_strides, axis,
                                      concat_dims, context);
    }
}

result<void> kernels::convert(datatype_t in_type, datatype_t out_type,
                              const gsl::byte *input, gsl::byte *output,
                              const runtime_shape_t &in_shape,
                              const runtime_shape_t &in_strides,
                              const runtime_shape_t &out_strides,
                              kernel_context &context) noexcept {
    return cpu::reference::convert(in_type, out_type, input, output, in_shape,
                                   in_strides, out_strides, context);
}

result<void> kernels::copy(datatype_t type, const gsl::byte *src,
                           gsl::byte *dest, const runtime_shape_t &shape,
                           const runtime_shape_t &src_strides,
                           const runtime_shape_t &dest_strides,
                           kernel_context &context) noexcept {
    const auto default_strides = get_default_strides(shape);
    auto src_dims_offset =
        get_last_not_contiguous_index(src_strides, default_strides);
    auto dest_dims_offset =
        get_last_not_contiguous_index(dest_strides, default_strides);

    auto src_continuous = src_strides == default_strides;
    auto dest_continuous = dest_strides == default_strides;
    if (!src_continuous && !dest_continuous) {
        return cpu::reference::copy(type, src, dest, shape, src_strides,
                                    dest_strides, context);
    } else {
        copy_impl_select select = copy_impl_select::all_contiguous;
        int dims_offset = -1;
        if (src_continuous && dest_continuous) {
            select = copy_impl_select::all_contiguous;
            dims_offset = -1;
        } else if (src_continuous && dest_dims_offset < 5) {
            select = copy_impl_select::src_contiguous;
            dims_offset = dest_dims_offset;
        } else if (dest_continuous && src_dims_offset < 5) {
            select = copy_impl_select::dest_contiguous;
            dims_offset = src_dims_offset;
        }
        return cpu::optimized::copy(type, src, dest, shape, src_strides,
                                    dest_strides, dims_offset, select, context);
    }
}

result<void> kernels::dequantize(datatype_t in_type, datatype_t out_type,
                                 const gsl::byte *input, gsl::byte *output,
                                 const runtime_shape_t &in_shape,
                                 const runtime_shape_t &in_strides,
                                 const runtime_shape_t &out_strides,
                                 float scale, float bias,
                                 kernel_context &context) noexcept {

    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(in_shape, out_strides)) {
        return cpu::optimized::dequantize(in_type, out_type, input, output,
                                          in_shape, in_strides, out_strides,
                                          scale, bias, context);
    }
    return cpu::reference::dequantize(in_type, out_type, input, output,
                                      in_shape, in_strides, out_strides, scale,
                                      bias, context);
}

template result<void> kernels::equal<uint8_t>(
    const uint8_t *input_a, const uint8_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::equal<float>(
    const float *input_a, const float *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_strides) noexcept;

template result<void> kernels::equal<int64_t>(
    const int64_t *input_a, const int64_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> kernels::equal(const T *input_a, const T *input_b, bool *output,
                            const runtime_shape_t &in_a_shape,
                            const runtime_shape_t &in_a_strides,
                            const runtime_shape_t &in_b_shape,
                            const runtime_shape_t &in_b_strides,
                            const runtime_shape_t &out_strides) noexcept {
    return cpu::reference::equal(input_a, input_b, output, in_a_shape,
                                 in_a_strides, in_b_shape, in_b_strides,
                                 out_strides);
}

result<void> kernels::lut1d(datatype_t type, const gsl::byte *input,
                            const gsl::byte *table, gsl::byte *output,
                            const runtime_shape_t &shape,
                            const runtime_shape_t &in_strides,
                            const runtime_shape_t &out_strides,
                            const scalar &min, const scalar &max) noexcept {
    return cpu::reference::lut1d(type, input, table, output, shape, in_strides,
                                 out_strides, min, max);
}

template result<void>
kernels::matmul<float>(const float *input_a, const float *input_b,
                       const float *bias, float *output,
                       const runtime_shape_t &in_a_shape,
                       const runtime_shape_t &in_b_shape,
                       value_range<float> fused_activation) noexcept;

template <typename T>
result<void> kernels::matmul(const T *input_a, const T *input_b, const T *bias,
                             T *output, const runtime_shape_t &in_a_shape,
                             const runtime_shape_t &in_b_shape,
                             value_range<float> fused_activation) noexcept {
    return cpu::reference::matmul(input_a, input_b, bias, output, in_a_shape,
                                  in_b_shape, fused_activation);
}

result<void>
kernels::onehot(datatype_t type, const int32_t *indices, gsl::byte *output,
                const runtime_shape_t &indices_shape,
                const runtime_shape_t &out_shape,
                const runtime_shape_t &out_strides, gsl::byte *depth,
                gsl::byte *off_value, gsl::byte *on_value, size_t axis,
                onehot_mode_t mode, kernel_context &context) noexcept {
    if (is_contiguous(out_shape, out_strides) &&
        (indices_shape.size() - axis) < 4) {
        return cpu::optimized::onehot(type, indices, output, indices_shape,
                                      out_shape, out_strides, depth, off_value,
                                      on_value, axis, mode, context);
    } else {
        return cpu::reference::onehot(type, indices, output, indices_shape,
                                      out_shape, out_strides, depth, off_value,
                                      on_value, axis, mode, context);
    }
}

result<void> kernels::pad(datatype_t type, const gsl::byte *input,
                          gsl::byte *output, const runtime_shape_t &in_shape,
                          const runtime_shape_t &in_strides,
                          const runtime_shape_t &out_strides,
                          const runtime_paddings_t &paddings, pad_mode_t mode,
                          const scalar &pad_value,
                          kernel_context &context) noexcept {
    return cpu::reference::pad(type, input, output, in_shape, in_strides,
                               out_strides, paddings, mode, pad_value, context);
}

result<void> kernels::quantize(datatype_t in_type, datatype_t out_type,
                               const gsl::byte *input, gsl::byte *output,
                               const runtime_shape_t &in_shape,
                               const runtime_shape_t &in_strides,
                               const runtime_shape_t &out_strides, float scale,
                               float bias, kernel_context &context) noexcept {
    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(in_shape, out_strides)) {
        return cpu::optimized::quantize(in_type, out_type, input, output,
                                        in_shape, in_strides, out_strides,
                                        scale, bias, context);
    }
    return cpu::reference::quantize(in_type, out_type, input, output, in_shape,
                                    in_strides, out_strides, scale, bias,
                                    context);
}

result<void> kernels::transpose(datatype_t type, const gsl::byte *src,
                                gsl::byte *dest,
                                const runtime_shape_t &in_shape,
                                const runtime_shape_t &perm,
                                const runtime_shape_t &in_strides,
                                const runtime_shape_t &out_strides,
                                kernel_context &context) noexcept {
    return cpu::reference::transpose(type, src, dest, in_shape, perm,
                                     in_strides, out_strides, context);
}

result<void> kernels::binary(
    binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_strides, value_range<float> fused_activation,
    kernel_context &context) noexcept {
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape,
                                  in_a_strides, in_b_shape, in_b_strides,
                                  out_strides, fused_activation, context);
}

result<void> kernels::unary(unary_op_t op, const float *input, float *output,
                            const runtime_shape_t &shape,
                            const runtime_shape_t &in_strides,
                            const runtime_shape_t &out_strides,
                            kernel_context &context) noexcept {
    return cpu::reference::unary(op, input, output, shape, in_strides,
                                 out_strides, context);
}

result<void> kernels::reduce(reduce_op_t op, float init_value,
                             const float *input, float *output,
                             const runtime_shape_t &in_shape,
                             const runtime_shape_t &axis,
                             const runtime_shape_t &in_strides,
                             const runtime_shape_t &out_strides, bool keep_dims,
                             kernel_context &context) noexcept {
    return cpu::reference::reduce(op, init_value, input, output, in_shape, axis,
                                  in_strides, out_strides, keep_dims, context);
}

template result<void> kernels::reduce_arg<int32_t>(
    reduce_arg_op_t op, const float *input, int32_t *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, const runtime_shape_t &axis,
    bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template result<void> kernels::reduce_arg<int64_t>(
    reduce_arg_op_t op, const float *input, int64_t *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, const runtime_shape_t &axis,
    bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template <typename T>
result<void> kernels::reduce_arg(reduce_arg_op_t op, const float *input,
                                 T *output, const runtime_shape_t &in_shape,
                                 const runtime_shape_t &in_strides,
                                 const runtime_shape_t &out_strides,
                                 const runtime_shape_t &axis, bool keep_dims,
                                 bool select_last_idx,
                                 kernel_context &context) noexcept {
    return cpu::reference::reduce_arg(op, input, output, in_shape, in_strides,
                                      out_strides, axis, keep_dims,
                                      select_last_idx, context);
}

template result<void> kernels::reduce_prod<float>(
    const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims) noexcept;

template <typename T>
result<void>
kernels::reduce_prod(const T *input, T *output, const runtime_shape_t &in_shape,
                     const runtime_shape_t &in_strides,
                     const runtime_shape_t &out_strides,
                     const runtime_shape_t &axes, bool keep_dims) noexcept {
    return cpu::reference::reduce_prod(input, output, in_shape, in_strides,
                                       out_strides, axes, keep_dims);
}

#define DISPATCH_RESIZE(resize_fun)                                            \
    runtime_shape_t out_shape{in_shape[0], in_shape[1],                        \
                              static_cast<size_t>(out_h),                      \
                              static_cast<size_t>(out_w)};                     \
    if (is_contiguous(in_shape, in_strides) &&                                 \
        is_contiguous(out_shape, out_strides)) {                               \
        return cpu::optimized::resize_fun(                                     \
            type, input, output, in_shape, in_strides, out_strides, out_h,     \
            out_w, align_corners, half_pixel_centers, context);                \
    } else {                                                                   \
        return cpu::reference::resize_fun(                                     \
            type, input, output, in_shape, in_strides, out_strides, out_h,     \
            out_w, align_corners, half_pixel_centers, context);                \
    }

result<void> kernels::resize_bilinear(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept {
    DISPATCH_RESIZE(resize_bilinear);
}

result<void> kernels::resize_nearest_neighbor(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w,
    bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept {
    DISPATCH_RESIZE(resize_nearest_neighbor);
}

result<void> kernels::slice(datatype_t type, const gsl::byte *input,
                            gsl::byte *output, const runtime_shape_t &in_shape,
                            const runtime_shape_t &in_strides,
                            const runtime_shape_t &out_strides,
                            const runtime_shape_t &begins,
                            const runtime_axis_t &ends,
                            const runtime_axis_t &strides,
                            kernel_context &context) noexcept {
    bool neg_strides = false;
    for (auto &&stride : strides) {
        if (stride < 0) {
            neg_strides = true;
            break;
        }
    }
    if (in_strides.size() <= 4 && !neg_strides) {
        return cpu::optimized::slice(type, input, output, in_shape, in_strides,
                                     out_strides, begins, ends, strides,
                                     context);
    } else {
        return cpu::reference::slice(type, input, output, in_shape, in_strides,
                                     out_strides, begins, ends, strides,
                                     context);
    }
}

result<void> kernels::gather(datatype_t in_type, const gsl::byte *input,
                             gsl::byte *output, const runtime_shape_t &in_shape,
                             const runtime_shape_t &out_shape,
                             const runtime_shape_t &in_strides,
                             const runtime_shape_t &out_strides,
                             const int32_t *indices,
                             const runtime_shape_t &indices_shape, int32_t axis,
                             kernel_context &context) noexcept {
    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(out_shape, out_strides)) {
        return cpu::optimized::gather(in_type, input, output, in_shape,
                                      out_shape, in_strides, out_strides,
                                      indices, indices_shape, axis, context);
    } else {
        return cpu::reference::gather(in_type, input, output, in_shape,
                                      out_shape, in_strides, out_strides,
                                      indices, indices_shape, axis, context);
    }
}

result<void> kernels::gather_nd(
    datatype_t in_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const int32_t *indices, const runtime_shape_t &indices_shape,
    int32_t batch_dims, kernel_context &context) noexcept {
    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(out_shape, out_strides)) {
        return cpu::optimized::gather_nd(
            in_type, input, output, in_shape, out_shape, in_strides,
            out_strides, indices, indices_shape, batch_dims, context);
    } else {
        return cpu::reference::gather_nd(
            in_type, input, output, in_shape, out_shape, in_strides,
            out_strides, indices, indices_shape, batch_dims, context);
    }
}

template result<void> kernels::cumsum<float>(const float *input, float *output,
                                             const runtime_shape_t &in_shape,
                                             int32_t axis, bool exclusive,
                                             bool reverse) noexcept;

template <typename T>
result<void> kernels::cumsum(const T *input, T *output,
                             const runtime_shape_t &in_shape, int32_t axis,
                             bool exclusive, bool reverse) noexcept {
    return cpu::reference::cumsum(input, output, in_shape, axis, exclusive,
                                  reverse);
}

template result<void> kernels::hardmax<float>(const float *input,
                                              const runtime_shape_t &in_shape,
                                              const runtime_shape_t &in_strides,
                                              float *output,
                                              int32_t axis) noexcept;

template <typename T>
result<void> kernels::hardmax(const T *input, const runtime_shape_t &in_shape,
                              const runtime_shape_t &in_strides, T *output,
                              int32_t axis) noexcept {
    return cpu::reference::hardmax(input, in_shape, in_strides, output, axis);
}

template result<void>
kernels::random_normal<float>(float *output, const runtime_shape_t &out_shape,
                              float mean, float std, float seed) noexcept;

template <typename T>
result<void> kernels::random_normal(T *output, const runtime_shape_t &out_shape,
                                    float mean, float std,
                                    float seed) noexcept {
    return cpu::reference::random_normal(output, out_shape, mean, std, seed);
}

template result<void>
kernels::random_uniform<float>(float *output, const runtime_shape_t &out_shape,
                               float low, float high, float seed) noexcept;

template <typename T>
result<void>
kernels::random_uniform(T *output, const runtime_shape_t &out_shape, float low,
                        float high, float seed) noexcept {
    return cpu::reference::random_uniform(output, out_shape, low, high, seed);
}

template result<void> kernels::roi_align<float>(
    const float *input, const float *rois, int64_t *batch_indices,
    float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, roi_align_mode_t mode,
    float spatial_scale, int64_t sampling_ratio) noexcept;

template <typename T>
result<void>
kernels::roi_align(const T *input, const T *rois, int64_t *batch_indices,
                   T *output, const runtime_shape_t &in_shape,
                   const runtime_shape_t &out_shape, roi_align_mode_t mode,
                   float spatial_scale, int64_t sampling_ratio) noexcept {
    return cpu::reference::roi_align(input, rois, batch_indices, output,
                                     in_shape, out_shape, mode, spatial_scale,
                                     sampling_ratio);
}

template result<void>
kernels::sigmoid<float>(const float *input, float *output,
                        const runtime_shape_t &in_shape,
                        const runtime_shape_t &in_strides) noexcept;

template <typename T>
result<void> kernels::sigmoid(const T *input, T *output,
                              const runtime_shape_t &in_shape,
                              const runtime_shape_t &in_strides) noexcept {
    return cpu::reference::sigmoid(input, output, in_shape, in_strides);
}

template result<void> kernels::ternary<float>(
    const float *input_a, const float *input_b, const float *input_c,
    float *output, const runtime_shape_t &in_a_shape,
    const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape,
    const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> kernels::ternary(
    const float *input_a, const T *input_b, const T *input_c, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept {
    return cpu::reference::ternary(
        input_a, input_b, input_c, output, in_a_shape, in_a_strides, in_b_shape,
        in_b_strides, in_c_shape, in_c_strides, out_strides);
}

template result<void> kernels::topk<float>(
    const float *input, float *output_values, int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &output_values_shape,
    const runtime_shape_t &output_values_strides,
    const runtime_shape_t &output_indices_shape,
    const runtime_shape_t &output_indices_strides, const int64_t k,
    const int32_t axis, const bool largest, const bool sorted) noexcept;

template <typename T>
result<void> kernels::topk(const T *input, T *output_values,
                           int64_t *output_indices,
                           const runtime_shape_t &in_shape,
                           const runtime_shape_t &in_strides,
                           const runtime_shape_t &output_values_shape,
                           const runtime_shape_t &output_values_strides,
                           const runtime_shape_t &output_indices_shape,
                           const runtime_shape_t &output_indices_strides,
                           const int64_t k, const int32_t axis,
                           const bool largest, const bool sorted) noexcept {
    return cpu::reference::topk(
        input, output_values, output_indices, in_shape, in_strides,
        output_values_shape, output_values_strides, output_indices_shape,
        output_indices_strides, k, axis, largest, sorted);
}

template result<void> kernels::trilu<float>(const float *input, float *output,
                                            const runtime_shape_t &in_shape,
                                            const bool upper,
                                            const int64_t k) noexcept;

template <typename T>
result<void> kernels::trilu(const T *input, T *output,
                            const runtime_shape_t &in_shape, const bool upper,
                            const int64_t k) noexcept {
    return cpu::reference::trilu(input, output, in_shape, upper, k);
}
