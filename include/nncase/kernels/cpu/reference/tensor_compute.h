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
#include "runtime_types.h"
#include <nncase/kernels/kernel_context.h>

BEGIN_NS_NNCASE_KERNELS_CPU_REF

NNCASE_API result<void> batch_to_space(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &crops, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    kernel_context &context) noexcept;

NNCASE_API result<void> broadcast(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, kernel_context &context) noexcept;

NNCASE_API result<void> concat(datatype_t type, gsl::span<const gsl::byte *const> inputs, gsl::byte *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims,
    kernel_context &context) noexcept;

NNCASE_API result<void> convert(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept;

NNCASE_API result<void> copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides, kernel_context &context) noexcept;

NNCASE_API result<void> transpose(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &perm, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept;

template <typename T>
NNCASE_API result<void> binary(binary_op_t op, const T *input_a, const T *input_b, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, value_range<float> fused_activation, kernel_context &context) noexcept;

NNCASE_API result<void> dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context) noexcept;

template <typename T>
NNCASE_API result<void> compare(compare_op_t op, const T *input_a, const T *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

NNCASE_API result<void> lut1d(datatype_t type, const gsl::byte *input, const gsl::byte *table, gsl::byte *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const scalar &min, const scalar &max) noexcept;

template <typename T>
NNCASE_API result<void> matmul(const T *input_a, const T *input_b, const T *bias, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept;

NNCASE_API result<void> onehot(datatype_t type, const int32_t *indices, gsl::byte *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, gsl::byte *depth, gsl::byte *off_value, gsl::byte *on_value, size_t axis, onehot_mode_t mode, kernel_context &context) noexcept;

NNCASE_API result<void> pad(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_paddings_t &paddings, pad_mode_t mode, const scalar &pad_value,
    kernel_context &context) noexcept;

NNCASE_API result<void> quantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias,
    kernel_context &context) noexcept;

NNCASE_API result<void> unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept;

template <typename T>
NNCASE_API result<void> reduce(reduce_op_t op, T init_value, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template <typename T>
NNCASE_API result<void> reduce_arg(reduce_arg_op_t op, const float *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axis, bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template <typename T>
result<void> reduce_prod(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims) noexcept;

NNCASE_API result<void> resize_bilinear(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept;

NNCASE_API result<void> resize_nearest_neighbor(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, int32_t out_h, int32_t out_w, bool align_corners, bool half_pixel_centers,
    kernel_context &context) noexcept;

NNCASE_API result<void> slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_axis_t &ends, const runtime_axis_t &strides,
    kernel_context &context) noexcept;

NNCASE_API result<void> gather(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, size_t axis, kernel_context &context) noexcept;

NNCASE_API result<void> gather_nd(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const int32_t *indices, const runtime_shape_t &indices_shape, size_t batch_dims, kernel_context &context) noexcept;

template <typename T>
NNCASE_API result<void> cumsum(const T *input, T *output, const runtime_shape_t &in_shape,
    int32_t axis, bool exclusive, bool reverse) noexcept;

template <typename T>
NNCASE_API result<void> hardmax(const T *input, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    T *output, int32_t axis) noexcept;

template <typename T>
NNCASE_API result<void> random_normal(T *output, const runtime_shape_t &out_shape, float mean, float std, float seed) noexcept;

template <typename T>
NNCASE_API result<void> random_uniform(T *output, const runtime_shape_t &out_shape, float low, float high, float seed) noexcept;

template <typename T>
NNCASE_API result<void> roi_align(const T *input, const T *rois, int64_t *batch_indices, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &out_shape, roi_align_mode_t mode, float spatial_scale, int64_t sampling_ratio) noexcept;

template <typename T>
NNCASE_API result<void> sigmoid(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept;

template <typename T>
NNCASE_API result<void> softmax(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides, int32_t axis, float beta) noexcept;

template <typename T>
NNCASE_API result<void> ternary(const float *input_a, const T *input_b, const T *input_c, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template <typename T>
NNCASE_API result<void> topk(const T *input, T *output_values, int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &output_values_shape, const runtime_shape_t &output_values_strides,
    const runtime_shape_t &output_indices_shape, const runtime_shape_t &output_indices_strides,
    const int64_t k, const int32_t axis, const bool largest, const bool sorted) noexcept;

template <typename T>
NNCASE_API result<void> trilu(const T *input, T *output, const runtime_shape_t &in_shape, const bool upper, const int64_t k) noexcept;

template <typename T>
NNCASE_API result<void> gru(const T *input, const T *w, const T *r, const T *b, T *initial_h, T *output, T *output_h, const runtime_shape_t &input_shape, const runtime_shape_t &w_shape, int mode) noexcept;

template <typename T>
NNCASE_API result<void> tflite_detection_postprocess(const T *boxes, const T *scores, const T *anchors, T *output_locations, T *output_classes, T *output_scores, T *output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale) noexcept;

NNCASE_API result<void> space_to_batch(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &block_shape, const runtime_paddings_t &paddings, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept;

template <typename TI, typename TK>
NNCASE_API result<void> gather_elements(const TI *input, const TK *indices, TI *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &indices_shape, const int axis) noexcept;

template <typename T>
NNCASE_API result<void> layernorm(const T *input, T *output, T *scale, T *bias, const runtime_shape_t &in_shape, int32_t axis, float epsilon) noexcept;

template <typename T>
NNCASE_API result<void> compress(const T *input, const uint8_t *condition, T *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis) noexcept;

END_NS_NNCASE_KERNELS_CPU_REF
