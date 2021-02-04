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
#include <nncase/kernels/cpu/reference/convolution.h>
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

result<void> reference::conv2d(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_channels = w_shape[0];
    const auto out_h = kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    runtime_shape_t in_index(4);
    runtime_shape_t w_index(4);
    runtime_shape_t bias_index(1);
    runtime_shape_t out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        in_index[0] = out_index[0] = batch;
        for (size_t og = 0; og < groups; og++)
        {
            for (size_t oc = 0; oc < g_oc; oc++)
            {
                out_index[1] = w_index[0] = bias_index[0] = og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++)
                {
                    out_index[2] = oy;
                    for (size_t ox = 0; ox < out_w; ox++)
                    {
                        out_index[3] = ox;
                        const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                        const size_t filter_y_start = (size_t)std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_x_start = (size_t)std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                        float value = bias[offset(bias_strides, bias_index)];

                        for (int32_t ic = 0; ic < g_ic; ic++)
                        {
                            in_index[1] = og * g_ic + ic;
                            w_index[1] = ic;
                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                w_index[2] = ky;
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    w_index[3] = kx;
                                    in_index[2] = in_y_origin + dilation_h * ky;
                                    in_index[3] = in_x_origin + dilation_w * kx;

                                    const float in_v = input[offset(in_strides, in_index)];
                                    const float w = weights[offset(w_strides, w_index)];

                                    value += in_v * w;
                                }
                            }
                        }

                        output[offset(out_strides, out_index)] = kernels::detail::apply_activation(value, fused_activation);
                    }
                }
            }
        }
    }

    return ok();
}
