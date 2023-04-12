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
#include <nncase/kernels/convolution.h>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/convolution.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;

result<void> kernels::conv2d(
    const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides,
    const padding &padding_h, const padding &padding_w, int32_t groups,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    value_range<float> fused_activation, kernel_context &context) noexcept {
    const auto batch = in_shape[0], in_h = in_shape[2], in_w = in_shape[3],
               out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(
        in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(
        in_w, filter_w, stride_w, dilation_w, padding_w);

    if (is_contiguous(in_shape, in_strides) &&
        is_contiguous(w_shape, w_strides) &&
        is_contiguous({batch, out_channels, out_h, out_w}, w_strides) &&
        dilation_h == 1 && dilation_w == 1) {
        if (cpu::optimized::conv2d(input, weights, bias, output, in_shape,
                                   in_strides, w_shape, w_strides, bias_strides,
                                   out_strides, padding_h, padding_w, groups,
                                   stride_h, stride_w, dilation_h, dilation_w,
                                   fused_activation, context)
                .is_ok()) {
            return ok();
        }
    }
    // general conv
    return cpu::reference::conv2d(
        input, weights, bias, output, in_shape, in_strides, w_shape, w_strides,
        bias_strides, out_strides, padding_h, padding_w, groups, stride_h,
        stride_w, dilation_h, dilation_w, fused_activation, context);
}
