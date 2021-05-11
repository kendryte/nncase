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

result<void> optimized::conv2d_1x1_s1(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto output_widths = in_shape[0] * w_shape[0] * in_shape[2] * in_shape[3];
    const auto widths = in_shape[2] * in_shape[3];
    // if oc's type is size_t, openmp will throw error in visual studio
    // if no cast, compiler will throw warning because of comparison of integer expressions of different signedness
    // warning be treated as errors
    const auto out_channels = static_cast<int>(w_shape[0]);
    // int threads = 1;
    // if (std::is_convertible_v<stackvm::stackvm_kernel_context &, decltype(context)>)
    // {
    //     threads = static_cast<stackvm::stackvm_kernel_context &>(context).num_threads_;
    // }
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        // #pragma omp parallel for num_threads(threads)
        // a img scan kernels/oc times
        for (int oc = 0; oc < out_channels; oc++)
        {
            const auto out_c = oc;
            const float *now_weights = weights + out_c * w_strides[0];
            const float *now_img_start = input + batch * in_strides[0];
            size_t channel = 0;
            // four channel in once for

            auto *now_output_channel_start = output + (batch * out_strides[0] + out_c * out_strides[1]);

            std::fill(now_output_channel_start, now_output_channel_start + in_shape[2] * in_shape[3], bias[oc]);
            for (; channel + 4 <= in_shape[1]; channel += 4, now_weights += 4)
            {
                auto *w_output = now_output_channel_start;
                // 1x1 w is constant in per channel
                const float w0 = now_weights[0];
                const float w1 = now_weights[1];
                const float w2 = now_weights[2];
                const float w3 = now_weights[3];

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
                }
            }

            for (; channel < in_shape[1]; ++channel)
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

result<void> optimized::conv2d_1x1_s2(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape,NNCASE_UNUSED  const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto batch = in_shape[0], in_channels = in_shape[1], in_h = in_shape[2], in_w = in_shape[3], out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding_w);

    const size_t tailstep = in_w - (out_w * stride_w);

    for (size_t b = 0; b < batch; b++)
    {
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++)
        {
            float *out = output + (b * out_channels * out_h * out_w) + (oc * out_h * out_w);

            std::fill(out, out + out_h * out_w, bias[oc]);
            size_t ic = 0;
            for (; ic + 3 < in_channels; ic += 4)
            {
                float *outptr = out;
                const float *img0 = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);
                const float *img1 = input + (b * in_channels * in_h * in_w) + ((ic + 1) * in_h * in_w);
                const float *img2 = input + (b * in_channels * in_h * in_w) + ((ic + 2) * in_h * in_w);
                const float *img3 = input + (b * in_channels * in_h * in_w) + ((ic + 3) * in_h * in_w);

                const float *r0 = img0;
                const float *r1 = img1;
                const float *r2 = img2;
                const float *r3 = img3;

                const float *k0 = weights + oc * in_channels + ic;
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
                const float *img0 = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);
                const float *kernel0 = weights + oc * in_channels + ic;
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

result<void> optimized::conv2d_3x3_s1(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape,NNCASE_UNUSED  const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto batch = in_shape[0], out_channels = w_shape[0], in_channels = w_shape[1], in_h = in_shape[2], in_w = in_shape[3];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding::zero());
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding::zero());
    for (size_t b = 0; b < batch; b++) // batch
    {
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++) // out channel
        {
            float *out = output + (b * out_channels * out_h * out_w) + (oc * out_h * out_w);

            std::fill(out, out + out_h * out_w, bias[oc]);

            for (size_t ic = 0; ic < in_channels; ic++) // in channel
            {
                float *outptr0 = out, *outptr1 = out + out_w;
                const float *img = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);
                const float *kernel = weights + oc * in_channels * 9 + ic * 9;
                const float *r0 = img;
                const float *r1 = img + in_w; // input width
                const float *r2 = img + (in_w * 2);
                const float *r3 = img + (in_w * 3);

                const float *k0 = kernel;
                const float *k1 = kernel + 3;
                const float *k2 = kernel + 6;

                size_t i = 0;
                for (; i + 1 < out_h; i += 2)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum0 = 0, sum1 = 0;
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
                        r0++, r1++, r2++, r3++, outptr0++, outptr1++;
                    }
                    // skip row
                    r0 += (in_w + 2);
                    r1 += (in_w + 2);
                    r2 += (in_w + 2);
                    r3 += (in_w + 2);
                    outptr0 += out_w;
                    outptr1 += out_w;
                }

                for (; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum0 = bias[oc];
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
                        r0++, r1++, r2++, outptr0++;
                    }
                    // skip row
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
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

result<void> optimized::conv2d_3x3_s2(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape,NNCASE_UNUSED  const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto batch = in_shape[0], in_channels = in_shape[1], in_h = in_shape[2], in_w = in_shape[3], out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding_w);

    const size_t tailstep = in_w - (out_w * stride_w);

    for (size_t b = 0; b < batch; b++)
    {
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++)
        {
            float *out = output + (b * out_channels * out_h * out_w) + (oc * out_h * out_w);

            std::fill(out, out + out_h * out_w, bias[oc]);

            for (size_t ic = 0; ic < in_channels; ic++)
            {
                float *outptr = out;

                const float *img = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);
                const float *kernel0 = weights + oc * in_channels * 9 + ic * 9;

                const float *r0 = img;
                const float *r1 = img + in_w;
                const float *r2 = img + in_w * 2;

                const float *k0 = kernel0;
                const float *k1 = kernel0 + 3;
                const float *k2 = kernel0 + 6;

                for (size_t i = 0; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];

                        *outptr += sum;

                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }

                    r0 += tailstep + in_w;
                    r1 += tailstep + in_w;
                    r2 += tailstep + in_w;
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

result<void> optimized::conv2d_5x5_s1(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape,NNCASE_UNUSED  const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{

    const auto batch = in_shape[0], in_channels = in_shape[1], in_h = in_shape[2], in_w = in_shape[3], out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding_w);

    for (size_t b = 0; b < batch; b++)
    {
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++)
        {
            float *out = output + (b * out_channels * out_h * out_w) + (oc * out_h * out_w);

            std::fill(out, out + out_h * out_w, bias[oc]);

            for (size_t ic = 0; ic < in_channels; ic++)
            {
                float *outptr = out;
                float *outptr2 = outptr + out_w;

                const float *img0 = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);

                const float *kernel0 = weights + oc * in_channels * 25 + ic * 25;

                const float *r0 = img0;
                const float *r1 = img0 + in_w;
                const float *r2 = img0 + in_w * 2;
                const float *r3 = img0 + in_w * 3;
                const float *r4 = img0 + in_w * 4;
                const float *r5 = img0 + in_w * 5;

                const float *k0 = kernel0;
                const float *k1 = kernel0 + 5;
                const float *k2 = kernel0 + 10;
                const float *k3 = kernel0 + 15;
                const float *k4 = kernel0 + 20;

                size_t i = 0;

                for (; i + 1 < out_h; i += 2)
                {
                    int remain = out_w;

                    for (; remain > 0; remain--)
                    {
                        float sum = 0;
                        float sum2 = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r0[3] * k0[3];
                        sum += r0[4] * k0[4];

                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r1[3] * k1[3];
                        sum += r1[4] * k1[4];

                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];
                        sum += r2[3] * k2[3];
                        sum += r2[4] * k2[4];

                        sum += r3[0] * k3[0];
                        sum += r3[1] * k3[1];
                        sum += r3[2] * k3[2];
                        sum += r3[3] * k3[3];
                        sum += r3[4] * k3[4];

                        sum += r4[0] * k4[0];
                        sum += r4[1] * k4[1];
                        sum += r4[2] * k4[2];
                        sum += r4[3] * k4[3];
                        sum += r4[4] * k4[4];

                        sum2 += r1[0] * k0[0];
                        sum2 += r1[1] * k0[1];
                        sum2 += r1[2] * k0[2];
                        sum2 += r1[3] * k0[3];
                        sum2 += r1[4] * k0[4];

                        sum2 += r2[0] * k1[0];
                        sum2 += r2[1] * k1[1];
                        sum2 += r2[2] * k1[2];
                        sum2 += r2[3] * k1[3];
                        sum2 += r2[4] * k1[4];

                        sum2 += r3[0] * k2[0];
                        sum2 += r3[1] * k2[1];
                        sum2 += r3[2] * k2[2];
                        sum2 += r3[3] * k2[3];
                        sum2 += r3[4] * k2[4];

                        sum2 += r4[0] * k3[0];
                        sum2 += r4[1] * k3[1];
                        sum2 += r4[2] * k3[2];
                        sum2 += r4[3] * k3[3];
                        sum2 += r4[4] * k3[4];

                        sum2 += r5[0] * k4[0];
                        sum2 += r5[1] * k4[1];
                        sum2 += r5[2] * k4[2];
                        sum2 += r5[3] * k4[3];
                        sum2 += r5[4] * k4[4];

                        *outptr += sum;
                        *outptr2 += sum2;

                        r0++;
                        r1++;
                        r2++;
                        r3++;
                        r4++;
                        r5++;
                        outptr++;
                        outptr2++;
                    }

                    r0 += 4 + in_w;
                    r1 += 4 + in_w;
                    r2 += 4 + in_w;
                    r3 += 4 + in_w;
                    r4 += 4 + in_w;
                    r5 += 4 + in_w;

                    outptr += out_w;
                    outptr2 += out_w;
                }

                for (; i < out_h; i++)
                {

                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r0[3] * k0[3];
                        sum += r0[4] * k0[4];

                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r1[3] * k1[3];
                        sum += r1[4] * k1[4];

                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];
                        sum += r2[3] * k2[3];
                        sum += r2[4] * k2[4];

                        sum += r3[0] * k3[0];
                        sum += r3[1] * k3[1];
                        sum += r3[2] * k3[2];
                        sum += r3[3] * k3[3];
                        sum += r3[4] * k3[4];

                        sum += r4[0] * k4[0];
                        sum += r4[1] * k4[1];
                        sum += r4[2] * k4[2];
                        sum += r4[3] * k4[3];
                        sum += r4[4] * k4[4];

                        *outptr += sum;

                        r0++;
                        r1++;
                        r2++;
                        r3++;
                        r4++;
                        outptr++;
                    }

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                }
            }
        }
    }
    for (size_t i = 0; i < batch * out_channels * out_h * out_w; ++i)
    {
        output[i] = kernels::detail::apply_activation(output[i], fused_activation);
    }
    return ok();
}

result<void> optimized::conv2d_5x5_s2(const float *input, const float *weights, const float *bias, float *output, const runtime_shape_t &in_shape,NNCASE_UNUSED  const runtime_shape_t &in_strides, const runtime_shape_t &w_shape, NNCASE_UNUSED const runtime_shape_t &w_strides, NNCASE_UNUSED const runtime_shape_t &bias_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED const padding &padding_h, NNCASE_UNUSED const padding &padding_w, NNCASE_UNUSED int32_t groups, NNCASE_UNUSED int32_t stride_h, NNCASE_UNUSED int32_t stride_w, NNCASE_UNUSED int32_t dilation_h, NNCASE_UNUSED int32_t dilation_w, value_range<float> fused_activation) noexcept
{
    const auto batch = in_shape[0], in_channels = in_shape[1], in_h = in_shape[2], in_w = in_shape[3], out_channels = w_shape[0];
    const auto filter_h = (int32_t)w_shape[2];
    const auto filter_w = (int32_t)w_shape[3];
    const auto out_h = kernels::detail::get_windowed_output_size(in_h, filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_w, filter_w, stride_w, dilation_w, padding_w);

    const size_t tailstep = in_w - (out_w * stride_w);

    for (size_t b = 0; b < batch; b++)
    {
        // #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t oc = 0; oc < out_channels; oc++)
        {
            float *out = output + (b * out_channels * out_h * out_w) + (oc * out_h * out_w);

            std::fill(out, out + out_h * out_w, bias[oc]);

            for (size_t ic = 0; ic < in_channels; ic++)
            {
                float *outptr = out;

                const float *img = input + (b * in_channels * in_h * in_w) + (ic * in_h * in_w);

                const float *kernel = weights + oc * in_channels * 25 + ic * 25;

                const float *r0 = img;
                const float *r1 = img + in_w;
                const float *r2 = img + in_w * 2;
                const float *r3 = img + in_w * 3;
                const float *r4 = img + in_w * 4;

                const float *k0 = kernel;
                const float *k1 = kernel + 5;
                const float *k2 = kernel + 10;
                const float *k3 = kernel + 15;
                const float *k4 = kernel + 20;

                for (size_t i = 0; i < out_h; i++)
                {
                    for (size_t remain = 0; remain < out_w; remain++)
                    {
                        float sum = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r0[3] * k0[3];
                        sum += r0[4] * k0[4];

                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r1[3] * k1[3];
                        sum += r1[4] * k1[4];

                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];
                        sum += r2[3] * k2[3];
                        sum += r2[4] * k2[4];

                        sum += r3[0] * k3[0];
                        sum += r3[1] * k3[1];
                        sum += r3[2] * k3[2];
                        sum += r3[3] * k3[3];
                        sum += r3[4] * k3[4];

                        sum += r4[0] * k4[0];
                        sum += r4[1] * k4[1];
                        sum += r4[2] * k4[2];
                        sum += r4[3] * k4[3];
                        sum += r4[4] * k4[4];

                        *outptr += sum;

                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        r3 += 2;
                        r4 += 2;
                        outptr++;
                    }

                    r0 += tailstep + in_w;
                    r1 += tailstep + in_w;
                    r2 += tailstep + in_w;
                    r3 += tailstep + in_w;
                    r4 += tailstep + in_w;
                }
            }
        }
    }
    for (size_t i = 0; i < batch * out_channels * out_h * out_w; ++i)
    {
        output[i] = kernels::detail::apply_activation(output[i], fused_activation);
    }
    return ok();
}