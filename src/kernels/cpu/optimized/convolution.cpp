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
#include <utility>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

result<void> optimized::conv2d_1x1_s1(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto output_widths = in_shape[0] * w_shape[0] * in_shape[2] * in_shape[3];
    const auto widths = in_shape[2] * in_shape[3];
    // if oc's type is size_t, openmp will throw error in visual studio
    // if no cast, compiler will throw warning because of comparison of integer expressions of different signedness
    // warning be treated as errors
    const auto out_channels = static_cast<int>(w_shape[0]);

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(GET_NUM_THREADS)
#endif
        for (int oc = 0; oc < static_cast<int>(out_channels); oc++)
        {
            const auto out_c = oc;
            const float *now_weights = weights + out_c * w_strides[0];
            const float *now_img_start = input + batch * in_strides[0];
            size_t channel = 0;

            auto *now_output_channel_start = output + (batch * out_strides[0] + out_c * out_strides[1]);

            std::fill(now_output_channel_start, now_output_channel_start + in_shape[2] * in_shape[3], bias[oc]);
            for (; channel + 4 <= in_shape[1]; channel += 4, now_weights += 4)
            {
                auto *w_output = now_output_channel_start;
                const float w0 = now_weights[0];
                const float w1 = now_weights[1];
                const float w2 = now_weights[2];
                const float w3 = now_weights[3];

                const float *i0 = now_img_start + (channel + 0) * in_strides[1];
                const float *i1 = now_img_start + (channel + 1) * in_strides[1];
                const float *i2 = now_img_start + (channel + 2) * in_strides[1];
                const float *i3 = now_img_start + (channel + 3) * in_strides[1];

                const float *v0 = i0;
                const float *v1 = i1;
                const float *v2 = i2;
                const float *v3 = i3;

                for (size_t index = 0; index < widths; ++index)
                {
                    float sum0 = *v0 * w0;
                    float sum1 = *v1 * w1;
                    float sum2 = *v2 * w2;
                    float sum3 = *v3 * w3;

                    *w_output += sum0 + sum1 + sum2 + sum3;

                    ++w_output;
                    ++v0;
                    ++v1;
                    ++v2;
                    ++v3;
                }
            }

            for (; channel < in_shape[1]; ++channel)
            {
                auto *w_output = now_output_channel_start;
                const float *v = now_img_start + channel * in_strides[1];
                for (size_t index = 0; index < widths; ++index)
                {
                    *w_output += (*now_weights) * (*v);
                    ++w_output;
                    ++v;
                }
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

result<void> optimized::conv2d_1x1_s2(const float *input, const float *weights, const float *bias, float *output,
    const runtime_shape_t &in_shape, NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &w_shape,
    NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w,
    NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w,
    NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation, NNCASE_UNUSED kernel_context &context) noexcept
{
    const auto batch = in_shape[0], in_channels = in_shape[1], in_h = in_shape[2], in_w = in_shape[3], out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding_w);

    const size_t tailstep = in_w - (out_w * stride_w);

    for (size_t b = 0; b < batch; b++)
    {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(GET_NUM_THREADS)
#endif
        for (int oc = 0; oc < static_cast<int>(out_channels); oc++)
        {
            float *out = output + (b * out_strides[0] + oc * out_strides[1]);

            std::fill(out, out + out_h * out_w, bias[oc]);
            size_t ic = 0;
            for (; ic + 3 < in_channels; ic += 4)
            {
                float *outptr = out;
                const float *img0 = input + (b * in_strides[0]) + (ic * in_strides[1]);
                const float *img1 = input + (b * in_strides[0]) + ((ic + 1) * in_strides[1]);
                const float *img2 = input + (b * in_strides[0]) + ((ic + 2) * in_strides[1]);
                const float *img3 = input + (b * in_strides[0]) + ((ic + 3) * in_strides[1]);

                const float *r0 = img0;
                const float *r1 = img1;
                const float *r2 = img2;
                const float *r3 = img3;

                const float *k0 = weights + oc * w_strides[0] + ic * w_strides[1];
                const float *k1 = k0 + 1;
                const float *k2 = k0 + 2;
                const float *k3 = k0 + 3;
                for (size_t i = 0; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        *outptr += r0[0] * k0[0];
                        *outptr += r1[0] * k1[0];
                        *outptr += r2[0] * k2[0];
                        *outptr += r3[0] * k3[0];
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        r3 += 2;
                        outptr++;
                    }
                    r0 += tailstep + in_w;
                    r1 += tailstep + in_w;
                    r2 += tailstep + in_w;
                    r3 += tailstep + in_w;
                }
            }

            for (; ic < in_channels; ic++)
            {
                float *outptr = out;                                              
                const float *img0 = input + (b * in_strides[0]) + (ic * in_strides[1]);
                const float *kernel0 = weights + oc * w_strides[0] + ic * w_strides[1];
                const float *r0 = img0;
                const float *k0 = kernel0;
                for (size_t i = 0; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        *outptr += r0[0] * k0[0];
                        r0 += 2;
                        outptr++;
                    }
                    r0 += tailstep + in_w;
                }
            }
        }
    }
    for (size_t _ = 0; _ < batch * out_channels * out_h * out_w; _++)
    {
        *(output + _) = kernels::detail::apply_activation(*(output + _), fused_activation);
    }
    return ok();
}