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

result<void> optimized::conv2d(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, const padding &padding_h, const padding &padding_w,
    int32_t groups, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation) noexcept
{

    return ok();
}

result<void> optimized::conv3x3s1_sse(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, const runtime_shape_t &w_strides,
    const runtime_shape_t &bias_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    const auto out_channels = w_shape[0], in_channels = in_shape[1];
    const auto out_h = kernels::detail::get_windowed_output_size(in_shape[2], 3, 1, 1, padding::zero());
    const auto out_w = kernels::detail::get_windowed_output_size(in_shape[3], 3, 1, 1, padding::zero());
    runtime_shape_t out_idx(4);
    runtime_shape_t in_idx(4);
    runtime_shape_t kernel_idx(4);
    for (size_t b = 0; b < in_shape[0]; b++) // batch
    {
        in_idx[0] = out_idx[0] = b;
        for (size_t p = 0; p < out_channels; p++) // out channel
        {
            kernel_idx[0] = out_idx[1] = p;
            float *out = &output[offset(out_strides, out_idx)];
            for (size_t q = 0; q < in_shape[1]; q++) // in channel
            {
                kernel_idx[1] = in_idx[1] = q;
                float *outptr0 = out, *outptr1 = out + out_w;
                const float *img = &input[offset(in_strides, in_idx)];
                const float *kernel = &weights[offset(w_strides, kernel_idx)];
                const float *r0 = img;
                const float *r1 = img + in_shape[3]; // input width
                const float *r2 = img + in_shape[3] * 2;
                const float *r3 = img + in_shape[3] * 3;

                const float *k0 = kernel;
                const float *k1 = kernel + 3;
                const float *k2 = kernel + (2 * 3);
                size_t i = 0;
                for (; i + 1 < out_h; i += 2)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum0 = bias[p], sum1 = bias[p];
                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k1[0];
                        sum0 += r1[1] * k1[1];
                        sum0 += r1[2] * k1[2];
                        sum0 += r2[0] * k2[0];
                        sum0 += r2[1] * k2[1];
                        sum0 += r2[2] * k2[2];

                        sum1 += r1[0] * k0[0];
                        sum1 += r1[1] * k0[1];
                        sum1 += r1[2] * k0[2];
                        sum1 += r2[0] * k1[0];
                        sum1 += r2[1] * k1[1];
                        sum1 += r2[2] * k1[2];
                        sum1 += r3[0] * k2[0];
                        sum1 += r3[1] * k2[1];
                        sum1 += r3[2] * k2[2];
                        *outptr0 += sum0;
                        *outptr1 += sum1;
                        *outptr0 = kernels::detail::apply_activation(*outptr0, fused_activation);
                        *outptr1 = kernels::detail::apply_activation(*outptr1, fused_activation);
                        r0++, r1++, r2++, r3++, outptr0++, outptr1++;
                    }
                    // skip row
                    r1 += (in_shape[3] + 2);
                    r2 += (in_shape[3] + 2);
                    r3 += (in_shape[3] + 2);
                    outptr0 += out_w;
                    outptr1 += out_w;
                }

                for (; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum0 = bias[p];
                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k1[0];
                        sum0 += r1[1] * k1[1];
                        sum0 += r1[2] * k1[2];
                        sum0 += r2[0] * k2[0];
                        sum0 += r2[1] * k2[1];
                        sum0 += r2[2] * k2[2];
                        *outptr0 += sum0;
                        *outptr0 = kernels::detail::apply_activation(*outptr0, fused_activation);
                        r0++, r1++, r2++, outptr0++;
                    }
                    // skip row
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0 += out_w;
                }
            }
        }
    }
    return ok();
}