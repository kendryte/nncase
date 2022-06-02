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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

namespace {
result<void>
conv2d_impl(const float *input, const float *weights, const float *bias,
            float *output, const dims_t &in_shape, const strides_t &in_strides,
            const dims_t &w_shape, const dims_t &w_strides,
            const dims_t &bias_strides, const strides_t &out_strides,
            const padding &padding_h, const padding &padding_w, int32_t groups,
            int32_t stride_h, int32_t stride_w, int32_t dilation_h,
            int32_t dilation_w, value_range<float> fused_activation,
            NNCASE_UNUSED kernel_context &context) noexcept {
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_channels = w_shape[0];
    const auto out_h = kernels::detail::get_windowed_output_size(
        in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(
        in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    dims_t in_index(4);
    dims_t w_index(4);
    dims_t bias_index(1);
    dims_t out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = out_index[0] = batch;
        for (size_t og = 0; og < (size_t)groups; og++) {
            for (size_t oc = 0; oc < g_oc; oc++) {
                out_index[1] = w_index[0] = bias_index[0] = og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++) {
                    out_index[2] = oy;
                    for (size_t ox = 0; ox < out_w; ox++) {
                        out_index[3] = ox;
                        const int32_t in_y_origin =
                            (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin =
                            (ox * stride_w) - padding_w.before;
                        const int32_t filter_y_start = (int32_t)std::max(
                            0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const int32_t filter_y_end = (int32_t)std::min(
                            filter_h, ((int32_t)in_shape[2] - in_y_origin +
                                       dilation_h - 1) /
                                          dilation_h);
                        const int32_t filter_x_start = (int32_t)std::max(
                            0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const int32_t filter_x_end = (int32_t)std::min(
                            filter_w, ((int32_t)in_shape[3] - in_x_origin +
                                       dilation_w - 1) /
                                          dilation_w);
                        float value = bias[offset(bias_strides, bias_index)];

                        for (size_t ic = 0; ic < g_ic; ic++) {
                            in_index[1] = og * g_ic + ic;
                            w_index[1] = ic;
                            for (int32_t ky = filter_y_start; ky < filter_y_end;
                                 ky++) {
                                w_index[2] = ky;
                                for (int32_t kx = filter_x_start;
                                     kx < filter_x_end; kx++) {
                                    w_index[3] = kx;
                                    in_index[2] = in_y_origin + dilation_h * ky;
                                    in_index[3] = in_x_origin + dilation_w * kx;

                                    const float in_v =
                                        input[offset(in_strides, in_index)];
                                    const float w =
                                        weights[offset(w_strides, w_index)];

                                    value += in_v * w;
                                }
                            }
                        }

                        output[offset(out_strides, out_index)] =
                            kernels::detail::apply_activation(value,
                                                              fused_activation);
                    }
                }
            }
        }
    }

    return ok();
}

dims_t infer_shape(const dims_t& in_shape, const dims_t& weights_shape, const dims_t& stride,
                   const dims_t& dilation, const paddings_t& paddings) {
    auto new_shape = in_shape;
    new_shape[2] = kernels::detail::get_windowed_output_size(in_shape[2], weights_shape[2], stride[0], dilation[0], paddings[0]);
    new_shape[3] = kernels::detail::get_windowed_output_size(in_shape[3], weights_shape[3], stride[1], dilation[1], paddings[1]);
    return new_shape;
}
} // namespace

result<value_t> nncase::kernels::stackvm::conv2d(
    [[maybe_unused]] pad_mode_t pad_mode, value_t input, value_t weights, value_t bias,
    value_t stride, value_t padding, value_t dilation, value_t groups,
    value_t fused_clamp, value_t output, kernel_context &context) {

    try_f32_input(input_mem, input);
    try_f32_input(weights_mem, weights);
    try_f32_input(bias_mem, bias);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_to_scalar(groups_value, groups, int32_t);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_array(fused_clamp_value, fused_clamp, float);
    auto out_shape = infer_shape(input_tensor->shape(), weights_tensor->shape(), strides_value, dilations, pads);
    try_f32_output(out_mem, output, input_tensor->dtype(), out_shape);
    try_(conv2d_impl(
        input_mem, weights_mem, bias_mem, out_mem, input_tensor->shape(),
        input_tensor->strides(), weights_tensor->shape(), weights_tensor->strides(), bias_tensor->strides(),
        output_tensor->strides(), pads[0], pads[1], groups_value, strides[0],
        strides[1], dilations[0], dilations[1],
        value_range<float>{fused_clamp_value[0], fused_clamp_value[1]},
        context));
    // todo: some param not be used
    return err(nncase_errc::runtime_not_found);
    return ok(output);
}
