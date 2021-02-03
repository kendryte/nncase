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
    const auto out_h = details::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = details::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_channels / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        const float *in_batch_p = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

        for (size_t og = 0; og < groups; og++)
        {
            const float *in_group_p = in_batch_p + (size_t)og * g_ic * in_shape[2] * in_shape[3];
            const float *w_group_p = weights + (size_t)og * g_oc * g_ic * filter_h * filter_w;

            for (size_t oc = 0; oc < g_oc; oc++)
            {
                const float *w_oc_p = w_group_p + (size_t)oc * g_ic * filter_h * filter_w;

                for (size_t oy = 0; oy < out_h; oy++)
                {
                    for (size_t ox = 0; ox < out_w; ox++)
                    {
                        const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                        const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                        const size_t filter_y_start = (size_t)std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_x_start = (size_t)std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                        const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                        float value = bias[og * g_oc + oc];

                        for (int32_t ic = 0; ic < g_ic; ic++)
                        {
                            const float *in_c_p = in_group_p + (size_t)ic * in_shape[2] * in_shape[3];
                            const float *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    const size_t in_y = in_y_origin + dilation_h * ky;
                                    const size_t in_x = in_x_origin + dilation_w * kx;

                                    const float in_v = in_c_p[in_y * in_shape[3] + in_x];
                                    const float w = w_ic_p[ky * filter_w + kx];

                                    value += in_v * w;
                                }
                            }
                        }

                        *output++ = details::apply_activation(value, fused_activation);
                    }
                }
            }
        }
    }

    return ok();
}
