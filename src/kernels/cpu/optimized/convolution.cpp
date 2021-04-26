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
#include <nncase/kernels/cpu/optimized/convolution.h>
#include <nncase/kernels/kernel_utils.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

result<void> optimized::conv2d_1x1(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto output_widths = in_shape[0] * w_shape[0] * in_shape[2] * in_shape[3];
    for (size_t i = 0; i < output_widths; ++i)
    {
        output[i] = 0;
    }
    const auto widths = in_shape[2] * in_shape[3];

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
#pragma omp parallel for 4
        // a img scan kernels/oc times
        for (size_t oc = 0; oc < w_shape[0]; oc++)
        {
            const float *now_weights = weights + oc * w_strides[0];
            const float *now_img_start = input + batch * in_strides[0];
            // Todo how bias fill
            size_t channel = 0;
            // four channel in once for

            auto *now_output_channel_start = output + (batch * out_strides[0] + oc * out_strides[1]);
            for (; channel + 4 <= in_shape[1]; channel += 4, now_weights += 4)
            {
                auto *w_output = now_output_channel_start;
                // 1x1 w is constant in per channel
                const float w0 =  now_weights[0];
                const float w1 =  now_weights[1];
                const float w2 =  now_weights[2];
                const float w3 =  now_weights[3];

                // Todo:incr index and incr pointer
                // reset address point to beginning of channel
                const float *i0 = now_img_start + (channel + 0) * in_strides[1];
                const float *i1 = now_img_start + (channel + 1) * in_strides[1];
                const float *i2 = now_img_start + (channel + 2) * in_strides[1];
                const float *i3 = now_img_start + (channel + 3) * in_strides[1];

                // four pointer point to pixel in four channel
                const float *v0 = i0;
                const float *v1 = i1;
                const float *v2 = i2;
                const float *v3 = i3;

                // treat 2 dimensions as linear 
                for (size_t index = 0; index < widths; ++index)
                {
                    float sum0 = *v0 * w0;
                    float sum1 = *v1 * w1;
                    float sum2 = *v2 * w2;
                    float sum3 = *v3 * w3;

                    *w_output += sum0 + sum1 + sum2 + sum3;

                    // move input pointer and output pointer
                    ++w_output;
                    // same as ++inputs
                    ++v0;
                    ++v1;
                    ++v2;
                    ++v3;
                    // const float v = input[batch * in_strides[0] + channel * in_strides[1] + index];
                }
            }

            for (;channel < in_shape[1]; ++channel)
            {
                // scan a img in per channel, the output will return start and add next channel value
                // reset write output pointer point to output channel start
                auto *w_output = now_output_channel_start;
                const float *v = now_img_start + channel * in_strides[1];
                for (size_t index = 0; index < widths; ++index)
                {
                    *w_output += (*now_weights) * (*v);
                    ++w_output;
                    ++v;
                }
                // same weight in a channel
                ++now_weights;
            }
        }
    }
    for (size_t i = 0; i < output_widths; ++i)
    {
        output[i] = kernels::detail::apply_activation(output[i], fused_activation);
    }
    return ok();
}