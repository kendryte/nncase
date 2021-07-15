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
#pragma once
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>

BEGIN_NS_NNCASE_KERNELS

NNCASE_API result<void> batch_to_space(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> broadcast(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> concat(datatype_t type, gsl::span<const gsl::byte *const> inputs, gsl::byte *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> convert(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> transpose(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> lut1d(datatype_t type, const gsl::byte *input, const gsl::byte *table, gsl::byte *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const scalar &min, const scalar &max) noexcept;

NNCASE_API result<void> onehot(datatype_t type, const int32_t *indices, gsl::byte *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, gsl::byte *depth, gsl::byte *off_value, gsl::byte *on_value, size_t axis, onehot_mode_t mode,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> pad(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, const scalar &pad_value,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> quantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> reduce(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> resize_bilinear(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> resize_nearest_neighbor(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
    kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> gather(datatype_t in_type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t axis, kernel_context &context = default_kernel_context()) noexcept;

NNCASE_API result<void> gather_nd(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, int32_t batch_dims, kernel_context &context = default_kernel_context()) noexcept;
END_NS_NNCASE_KERNELS
