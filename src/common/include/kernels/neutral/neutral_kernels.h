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
#pragma once
#include "../kernel_utils.h"
#include <cmath>
#include <runtime/nnil.h>
#include <runtime/runtime_op_utility.h>
#include <xtl/xspan.hpp>
#ifdef __riscv
#include "../riscv/neutral_kernels.h"
#endif

namespace nncase
{
namespace kernels
{
    namespace neutral
    {
        template <class TOp>
        void binary(const float *input_a, const float *input_b, float *output, const runtime_shape_t &in_a_shape,
            const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape, const value_range<float> &fused_activation, TOp &&op)
        {
            // opt. no broadcast
            if (in_a_shape == in_b_shape)
            {
                auto size = kernels::details::compute_size(in_a_shape);
                for (size_t i = 0; i < size; i++)
                {
                    const auto a = input_a[i];
                    const auto b = input_b[i];
                    output[i] = kernels::details::apply_activation(op(a, b), fused_activation);
                }
            }
            // fallback
            else
            {
                for (int32_t d0 = 0; d0 < out_shape[0]; d0++)
                {
                    for (int32_t d1 = 0; d1 < out_shape[1]; d1++)
                    {
                        for (int32_t d2 = 0; d2 < out_shape[2]; d2++)
                        {
                            for (int32_t d3 = 0; d3 < out_shape[3]; d3++)
                            {
                                runtime_shape_t in_off = { d0, d1, d2, d3 };
                                const auto in_a_off = kernels::details::get_reduced_offset(in_off, in_a_shape);
                                const auto in_b_off = kernels::details::get_reduced_offset(in_off, in_b_shape);
                                const auto a = input_a[offset(in_a_shape, in_a_off)];
                                const auto b = input_b[offset(in_b_shape, in_b_off)];
                                output[offset(out_shape, in_off)] = kernels::details::apply_activation(op(a, b), fused_activation);
                            }
                        }
                    }
                }
            }
        }

        template <class TOp>
        void quantized_binary(const uint8_t *input_a, const uint8_t *input_b, uint8_t *output, const runtime_shape_t &in_a_shape,
            const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape, int32_t input_a_offset, int32_t input_a_mul, int32_t input_a_shift,
            int32_t input_b_offset, int32_t input_b_mul, int32_t input_b_shift, int32_t output_mul, int32_t output_shift, int32_t output_offset, TOp &&op)
        {
            // opt. no broadcast
            if (in_a_shape == in_b_shape)
            {
                auto size = kernels::details::compute_size(in_a_shape);
                for (size_t i = 0; i < size; i++)
                {
                    auto a = (int32_t)input_a[i];
                    auto b = (int32_t)input_b[i];
                    a = runtime::mul_and_carry_shift(a + input_a_offset, input_a_mul, input_a_shift);
                    b = runtime::mul_and_carry_shift(b + input_b_offset, input_b_mul, input_b_shift);

                    auto output_val = runtime::mul_and_carry_shift(op(a, b), output_mul, output_shift);
                    output[i] = (uint8_t)std::clamp(output_val + output_offset, 0, 255);
                }
            }
            // fallback
            else
            {
                for (int32_t d0 = 0; d0 < out_shape[0]; d0++)
                {
                    for (int32_t d1 = 0; d1 < out_shape[1]; d1++)
                    {
                        for (int32_t d2 = 0; d2 < out_shape[2]; d2++)
                        {
                            for (int32_t d3 = 0; d3 < out_shape[3]; d3++)
                            {
                                runtime_shape_t in_off = { d0, d1, d2, d3 };
                                const auto in_a_off = kernels::details::get_reduced_offset(in_off, in_a_shape);
                                const auto in_b_off = kernels::details::get_reduced_offset(in_off, in_b_shape);
                                auto a = (int32_t)input_a[offset(in_a_shape, in_a_off)];
                                auto b = (int32_t)input_b[offset(in_b_shape, in_b_off)];
                                a = runtime::mul_and_carry_shift(a + input_a_offset, input_a_mul, input_a_shift);
                                b = runtime::mul_and_carry_shift(b + input_b_offset, input_b_mul, input_b_shift);

                                auto output_val = runtime::mul_and_carry_shift(op(a, b), output_mul, output_shift);
                                output[offset(out_shape, in_off)] = (uint8_t)std::clamp(output_val + output_offset, 0, 255);
                            }
                        }
                    }
                }
            }
        }

        template <class TRange, class TPtrGetter = details::default_ptr_getter<uint8_t, TRange>>
        inline void concat(xtl::span<TRange> inputs, uint8_t *output, xtl::span<const int32_t> concat_dims, size_t inner_size, size_t outer_size, TPtrGetter getter = {})
        {
            for (size_t oc = 0; oc < outer_size; oc++)
            {
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    auto size = inner_size * concat_dims[i];
                    auto src = getter(inputs[i]) + oc * size;
                    std::copy(src, src + size, output);
                    output += size;
                }
            }
        }

        inline void conv2d(const float *input, float *output, const float *weights, const float *bias, const runtime_shape_t &in_shape,
            int32_t groups, int32_t out_channels, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
            const padding &padding_h, const padding &padding_w, const value_range<float> &fused_activation)
        {
            const auto out_h = details::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
            const auto out_w = details::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
            const auto g_ic = in_shape[1] / groups;
            const auto g_oc = out_channels / groups;

            for (int32_t batch = 0; batch < in_shape[0]; batch++)
            {
                const float *in_batch_p = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

                for (int32_t og = 0; og < groups; og++)
                {
                    const float *in_group_p = in_batch_p + (size_t)og * g_ic * in_shape[2] * in_shape[3];
                    const float *w_group_p = weights + (size_t)og * g_oc * g_ic * filter_h * filter_w;

                    for (int32_t oc = 0; oc < g_oc; oc++)
                    {
                        const float *w_oc_p = w_group_p + (size_t)oc * g_ic * filter_h * filter_w;

                        for (int32_t oy = 0; oy < out_h; oy++)
                        {
                            for (int32_t ox = 0; ox < out_w; ox++)
                            {
                                const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                                const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                                const int32_t filter_y_start = std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_y_end = std::min(filter_h, (in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_x_start = std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                                const int32_t filter_x_end = std::min(filter_w, (in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                                float value = bias[og * g_oc + oc];

                                for (int32_t ic = 0; ic < g_ic; ic++)
                                {
                                    const float *in_c_p = in_group_p + (size_t)ic * in_shape[2] * in_shape[3];
                                    const float *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                                    for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                                    {
                                        for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                        {
                                            const int32_t in_y = in_y_origin + dilation_h * ky;
                                            const int32_t in_x = in_x_origin + dilation_w * kx;

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
        }

        inline void quantized_conv2d(const uint8_t *input, uint8_t *output, const uint8_t *weights, const int32_t *bias, int32_t input_offset, int32_t filter_offset,
            int32_t output_mul, int32_t output_shift, int32_t output_offset, const runtime_shape_t &in_shape, int32_t groups, int32_t out_channels,
            int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
            const padding &padding_h, const padding &padding_w)
        {
            const auto out_h = details::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
            const auto out_w = details::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
            const auto g_ic = in_shape[1] / groups;
            const auto g_oc = out_channels / groups;

            for (int32_t batch = 0; batch < in_shape[0]; batch++)
            {
                const uint8_t *in_batch_p = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

                for (int32_t og = 0; og < groups; og++)
                {
                    const uint8_t *in_group_p = in_batch_p + (size_t)og * g_ic * in_shape[2] * in_shape[3];
                    const uint8_t *w_group_p = weights + (size_t)og * g_oc * g_ic * filter_h * filter_w;

                    for (int32_t oc = 0; oc < g_oc; oc++)
                    {
                        const uint8_t *w_oc_p = w_group_p + (size_t)oc * g_ic * filter_h * filter_w;

                        for (int32_t oy = 0; oy < out_h; oy++)
                        {
                            for (int32_t ox = 0; ox < out_w; ox++)
                            {
                                const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                                const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                                const int32_t filter_y_start = std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_y_end = std::min(filter_h, (in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_x_start = std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                                const int32_t filter_x_end = std::min(filter_w, (in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                                int32_t value = bias[og * g_oc + oc];

                                for (int32_t ic = 0; ic < g_ic; ic++)
                                {
                                    const uint8_t *in_c_p = in_group_p + (size_t)ic * in_shape[2] * in_shape[3];
                                    const uint8_t *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                                    for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                                    {
                                        for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                        {
                                            const int32_t in_y = in_y_origin + dilation_h * ky;
                                            const int32_t in_x = in_x_origin + dilation_w * kx;

                                            const int32_t in_v = (int32_t)in_c_p[in_y * in_shape[3] + in_x] + input_offset;
                                            const int32_t w = (int32_t)w_ic_p[ky * filter_w + kx] + filter_offset;

                                            value += in_v * w;
                                        }
                                    }
                                }

                                auto output_val = static_cast<int32_t>(runtime::mul_and_carry_shift(value, output_mul, output_shift));
                                output_val += output_offset;
                                *output++ = (uint8_t)std::clamp(output_val, 0, 255);
                            }
                        }
                    }
                }
            }
        }

        inline void conv2d_transpose(const float *input, float *output, const float *weights, const float *bias, const runtime_shape_t &in_shape,
            int32_t groups, const runtime_shape_t &out_shape, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
            const padding &padding_h, const padding &padding_w, const value_range<float> &fused_activation)
        {
            std::fill(output, output + kernels::details::compute_size(out_shape), 0.f);
            const auto g_ic = in_shape[1] / groups;
            const auto g_oc = out_shape[1] / groups;

            for (int32_t batch = 0; batch < in_shape[0]; batch++)
            {
                float *out_batch_p = output + (size_t)batch * out_shape[1] * out_shape[2] * out_shape[3];

                for (int32_t g = 0; g < groups; g++)
                {
                    float *out_group_p = out_batch_p + (size_t)g * g_oc * out_shape[2] * out_shape[3];
                    const float *w_group_p = weights + (size_t)g * g_oc * g_ic * filter_h * filter_w;

                    for (int32_t ic = 0; ic < g_ic; ic++)
                    {
                        for (int32_t iy = 0; iy < in_shape[2]; iy++)
                        {
                            for (int32_t ix = 0; ix < in_shape[3]; ix++)
                            {
                                const int32_t out_y_origin = (iy * stride_h) - padding_h.before;
                                const int32_t out_x_origin = (ix * stride_w) - padding_w.before;
                                const int32_t filter_y_start = std::max(0, (-out_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_y_end = std::min(filter_h, (out_shape[2] - out_y_origin + dilation_h - 1) / dilation_h);
                                const int32_t filter_x_start = std::max(0, (-out_x_origin + dilation_w - 1) / dilation_w);
                                const int32_t filter_x_end = std::min(filter_w, (out_shape[3] - out_x_origin + dilation_w - 1) / dilation_w);
                                const float in_v = *input++;

                                for (int32_t oc = 0; oc < g_oc; oc++)
                                {
                                    float value = bias[g * g_oc + oc];
                                    float *out_c_p = out_group_p + (size_t)oc * out_shape[2] * out_shape[3];
                                    const float *w_oc_p = w_group_p + (size_t)oc * g_ic * filter_h * filter_w;
                                    const float *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                                    for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                                    {
                                        for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                        {
                                            const int32_t out_y = out_y_origin + dilation_h * ky;
                                            const int32_t out_x = out_x_origin + dilation_w * kx;

                                            const float w = w_ic_p[ky * filter_w + kx];

                                            out_c_p[out_y * out_shape[3] + out_x] += in_v * w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (fused_activation != value_range<float>::full())
            {
                for (size_t i = 0; i < kernels::details::compute_size(out_shape); i++)
                    output[i] = details::apply_activation(output[i], fused_activation);
            }
        }

        template <class TQ>
        void dequantize(const TQ *CXX_RESTRICT input, float *CXX_RESTRICT output, size_t count, const quant_param_t &param)
        {
#if __riscv
            riscv_dequantize(input, output, count, param);
#else
            float div = 1.f / param.scale;

            for (size_t i = 0; i < count; i++)
            {
                output[i] = (input[i] - param.zero_point) * div;
            }
#endif
        }

        inline void matmul(const float *input_a, const float *input_b, float *output, const float *bias, int32_t a_rows, int32_t a_cols, int32_t b_cols, const value_range<float> &fused_activation)
        {
            for (size_t oy = 0; oy < a_rows; oy++)
            {
                for (size_t ox = 0; ox < b_cols; ox++)
                {
                    float value = bias[ox];

                    for (size_t i = 0; i < a_cols; i++)
                    {
                        const auto a = input_a[oy * a_cols + i];
                        const auto b = input_b[i * b_cols + ox];
                        value += a * b;
                    }

                    output[oy * b_cols + ox] = details::apply_activation(value, fused_activation);
                }
            }
        }

        inline void quantized_matmul(const uint8_t *input_a, const uint8_t *input_b, uint8_t *output, const int32_t *bias, int32_t a_rows, int32_t a_cols, int32_t b_cols, int32_t input_a_offset, int32_t input_b_offset,
            int32_t output_mul, int32_t output_shift, int32_t output_offset)
        {
            for (size_t oy = 0; oy < a_rows; oy++)
            {
                for (size_t ox = 0; ox < b_cols; ox++)
                {
                    int32_t value = bias[ox];
                    for (size_t i = 0; i < a_cols; i++)
                    {
                        const auto a = (int32_t)input_a[oy * a_cols + i] + input_a_offset;
                        const auto b = (int32_t)input_b[i * b_cols + ox] + input_b_offset;
                        value += a * b;
                    }

                    auto output_val = static_cast<int32_t>(runtime::mul_and_carry_shift(value, output_mul, output_shift));
                    output_val += output_offset;
                    output[oy * b_cols + ox] = (uint8_t)std::clamp(output_val, 0, 255);
                }
            }
        }

        template <class T>
        void pad(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_paddings_t &paddings, T pad_value)
        {
            runtime_shape_t out_shape = { in_shape[0] + paddings[0].sum(),
                in_shape[1] + paddings[1].sum(),
                in_shape[2] + paddings[2].sum(),
                in_shape[3] + paddings[3].sum() };

            for (int d0 = 0; d0 < out_shape[0]; d0++)
            {
                auto d0_origin = -paddings[0].before;
                auto in0 = input + ((size_t)d0_origin + d0) * in_shape[1] * in_shape[2] * in_shape[3];

                for (int d1 = 0; d1 < out_shape[1]; d1++)
                {
                    auto d1_origin = -paddings[1].before;
                    auto in1 = in0 + ((size_t)d1_origin + d1) * in_shape[2] * in_shape[3];

                    for (int d2 = 0; d2 < out_shape[2]; d2++)
                    {
                        auto d2_origin = -paddings[2].before;
                        auto in2 = in1 + ((size_t)d2_origin + d2) * in_shape[3];

                        for (int d3 = 0; d3 < out_shape[3]; d3++)
                        {
                            auto d3_origin = -paddings[3].before;

                            if (d0 < paddings[0].before || d0 >= out_shape[0] - paddings[0].after
                                || d1 < paddings[1].before || d1 >= out_shape[1] - paddings[1].after
                                || d2 < paddings[2].before || d2 >= out_shape[2] - paddings[2].after
                                || d3 < paddings[3].before || d3 >= out_shape[3] - paddings[3].after)
                                *output++ = pad_value;
                            else
                                *output++ = in2[d3_origin + d3];
                        }
                    }
                }
            }
        }

        template <class TQ>
        void quantize(const float *CXX_RESTRICT input, TQ *CXX_RESTRICT output, size_t count, const quant_param_t &param)
        {
#if __riscv
            riscv_quantize(input, output, count, param);
#else
            for (size_t i = 0; i < count; i++)
            {
                int32_t tmp = (int32_t)roundf(input[i] * param.scale + param.zero_point);
                output[i] = std::clamp(tmp, (int32_t)std::numeric_limits<TQ>::lowest(), (int32_t)std::numeric_limits<TQ>::max());
            }
#endif
        }

        template <class TReducer>
        void reduce(const float *input, float *output, float init_value, const runtime_shape_t &in_shape, const runtime_shape_t &reduced_shape, TReducer &&reducer)
        {
            std::fill(output, output + kernels::details::compute_size(reduced_shape), init_value);

            for (int32_t d0 = 0; d0 < in_shape[0]; d0++)
            {
                for (int32_t d1 = 0; d1 < in_shape[1]; d1++)
                {
                    for (int32_t d2 = 0; d2 < in_shape[2]; d2++)
                    {
                        for (int32_t d3 = 0; d3 < in_shape[3]; d3++)
                        {
                            runtime_shape_t in_off = { d0, d1, d2, d3 };
                            auto out_off = kernels::details::get_reduced_offset(in_off, reduced_shape);
                            const auto a = input[offset(in_shape, in_off)];
                            auto &b = output[offset(reduced_shape, out_off)];
                            b = reducer(b, a);
                        }
                    }
                }
            }
        }

        template <class TOp>
        void unary(const float *CXX_RESTRICT input, float *CXX_RESTRICT output, size_t count, TOp &&op)
        {
            for (size_t i = 0; i < count; i++)
                output[i] = op(input[i]);
        }

        template <class TBinaryOp, class TOutputOp>
        void reduce_window2d(const float *input, float *output, float init_value, const runtime_shape_t &in_shape, int32_t filter_h, int32_t filter_w,
            int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, const padding &padding_h, const padding &padding_w,
            const value_range<float> &fused_activation, TBinaryOp &&binary_op, TOutputOp &&window_op)
        {
            const auto out_h = kernels::details::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
            const auto out_w = kernels::details::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
            runtime_shape_t out_shape { in_shape[0], in_shape[1], out_h, out_w };

            for (int32_t batch = 0; batch < in_shape[0]; batch++)
            {
                for (int32_t oc = 0; oc < in_shape[1]; oc++)
                {
                    for (int32_t oy = 0; oy < out_h; oy++)
                    {
                        for (int32_t ox = 0; ox < out_w; ox++)
                        {
                            const int32_t in_y_origin = (oy * stride_h) - padding_h.before;
                            const int32_t in_x_origin = (ox * stride_w) - padding_w.before;
                            const int32_t filter_y_start = std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                            const int32_t filter_y_end = std::min(filter_h, (in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                            const int32_t filter_x_start = std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                            const int32_t filter_x_end = std::min(filter_w, (in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                            float value = init_value;
                            int32_t kernel_count = 0;

                            for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    const int32_t in_y = in_y_origin + dilation_h * ky;
                                    const int32_t in_x = in_x_origin + dilation_w * kx;

                                    const float in_v = input[offset(in_shape, { batch, oc, in_y, in_x })];

                                    value = binary_op(value, in_v);
                                    kernel_count++;
                                }
                            }

                            output[offset(out_shape, { batch, oc, oy, ox })] = kernels::details::apply_activation(window_op(value, kernel_count), fused_activation);
                        }
                    }
                }
            }
        }

        template <class T>
        void resize_nearest_neighbor(const T *input, T *output, const runtime_shape_t &in_shape, int32_t out_h, int32_t out_w)
        {
            auto height_scale = (float)in_shape[2] / out_h;
            auto width_scale = (float)in_shape[3] / out_w;

            for (int batch = 0; batch < in_shape[0]; batch++)
            {
                auto in_batch = input + batch * in_shape[1] * in_shape[2] * in_shape[3];

                for (int oc = 0; oc < in_shape[1]; oc++)
                {
                    auto in_c = in_batch + oc * in_shape[2] * in_shape[3];

                    for (int oy = 0; oy < out_h; oy++)
                    {
                        auto in_y = std::min((int32_t)floorf(oy * height_scale), in_shape[2] - 1);
                        auto in_row = in_c + in_y * in_shape[3];

                        for (int ox = 0; ox < out_w; ox++)
                        {
                            auto in_x = std::min((int32_t)floorf(ox * width_scale), in_shape[3] - 1);
                            *output++ = in_row[in_x];
                        }
                    }
                }
            }
        }

        template <class T>
        inline void resize_bilinear(const T *input, T *output, const runtime_shape_t &in_shape, int32_t out_h, int32_t out_w, bool align_corners)
        {
            auto height_scale = (float)in_shape[2] / out_h;
            auto width_scale = (float)in_shape[3] / out_w;
            if (align_corners && out_h > 1)
                height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
            if (align_corners && out_w > 1)
                width_scale = (float)(in_shape[3] - 1) / (out_w - 1);

            auto destIdx = 0;
            for (int batch = 0; batch < in_shape[0]; batch++)
            {
                auto in_batch = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

                for (int oc = 0; oc < in_shape[1]; oc++)
                {
                    auto in_c = in_batch + (size_t)oc * in_shape[2] * in_shape[3];

                    for (int oy = 0; oy < out_h; oy++)
                    {
                        auto in_y = oy * height_scale;
                        auto in_y0 = (int)floorf(in_y);
                        auto in_y1 = std::min(in_y0 + 1, in_shape[2] - 1);

                        for (int ox = 0; ox < out_w; ox++)
                        {
                            auto in_x = ox * width_scale;
                            auto in_x0 = (int)floorf(in_x);
                            auto in_x1 = std::min(in_x0 + 1, in_shape[3] - 1);

                            auto v0 = in_c[in_y0 * in_shape[3] + in_x0];
                            auto v1 = in_c[in_y1 * in_shape[3] + in_x0];
                            auto v2 = in_c[in_y0 * in_shape[3] + in_x1];
                            auto v3 = in_c[in_y1 * in_shape[3] + in_x1];

                            auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                            auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                            auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                            auto a3 = (in_y - in_y0) * (in_x - in_x0);

                            output[destIdx++] = T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3);
                        }
                    }
                }
            }
        }

        inline void softmax(const float *input, float *output, float beta, int32_t outer_size, size_t inner_size)
        {
            for (size_t batch = 0; batch < outer_size; batch++)
            {
                auto src = input + batch * inner_size;
                auto dest = output + batch * inner_size;

                auto max = *std::max_element(src, src + inner_size);
                float sum = 0;

                for (size_t i = 0; i < inner_size; i++)
                {
                    auto value = expf((src[i] - max) * beta);
                    sum += value;
                    dest[i] = value;
                }

                for (size_t i = 0; i < inner_size; i++)
                    dest[i] /= sum;
            }
        }

        template <class T>
        void transpose(const T *CXX_RESTRICT input, T *CXX_RESTRICT output, const runtime_shape_t &in_shape, const runtime_shape_t &perm)
        {
            runtime_shape_t out_shape;
            for (size_t i = 0; i < 4; i++)
                out_shape[i] = in_shape[perm[i]];

            runtime_shape_t i, o;
            for (o[3] = 0; o[3] < out_shape[3]; o[3]++)
            {
                i[perm[3]] = o[3];
                for (o[2] = 0; o[2] < out_shape[2]; o[2]++)
                {
                    i[perm[2]] = o[2];
                    for (o[1] = 0; o[1] < out_shape[1]; o[1]++)
                    {
                        i[perm[1]] = o[1];
                        for (o[0] = 0; o[0] < out_shape[0]; o[0]++)
                        {
                            i[perm[0]] = o[0];
                            output[offset(out_shape, o)] = input[offset(in_shape, i)];
                        }
                    }
                }
            }
        }

        template <class T>
        void strided_slice(const T *CXX_RESTRICT input, T *CXX_RESTRICT output, const runtime_shape_t &in_shape, const runtime_shape_t &begin, const runtime_shape_t &end, const runtime_shape_t &strides)
        {
            auto loop_cond = [](int32_t i, int32_t stop, int32_t stride) {
                return stride > 0 ? i < stop : i > stop;
            };

            for (int32_t d0 = begin[0]; loop_cond(d0, end[0], strides[0]); d0 += strides[0])
            {
                auto d0_origin = input + (size_t)d0 * in_shape[1] * in_shape[2] * in_shape[3];
                for (int d1 = begin[1]; loop_cond(d1, end[1], strides[1]); d1 += strides[1])
                {
                    auto d1_origin = d0_origin + (size_t)d1 * in_shape[2] * in_shape[3];
                    for (int32_t d2 = begin[2]; loop_cond(d2, end[2], strides[2]); d2 += strides[2])
                    {
                        auto d2_origin = d1_origin + (size_t)d2 * in_shape[3];
                        for (int32_t d3 = begin[3]; loop_cond(d3, end[3], strides[3]); d3 += strides[3])
                            *output++ = d2_origin[d3];
                    }
                }
            }
        }

        inline void nnil_unary_method(const float *input, float *output, size_t count, xtl::span<const uint8_t> body)
        {
            using namespace nncase::runtime;

            for (size_t i = 0; i < count; i++)
            {
                nnil_evalstack stack;
                span_reader sr(body);
                nnil_reader reader(sr);
                bool ret = false;

                while (reader.avail() && !ret)
                {
                    auto op = reader.next();
                    switch (op.opcode)
                    {
                    case nnil_nop:
                        break;
                    case nnil_dup:
                        stack.dup();
                        break;
                    case nnil_pop:
                        stack.pop();
                        break;
                    case nnil_lda_0:
                        stack.push(input[i]);
                        break;
                    case nnil_ldc_r4_0:
                        stack.push(0.f);
                        break;
                    case nnil_ldc_r4_1:
                        stack.push(1.f);
                        break;
                    case nnil_ldc_r4:
                        stack.push(op.ldc_r4.r4);
                        break;
                    case nnil_abs:
                        stack.push(fabsf(stack.pop()));
                        break;
                    case nnil_ceil:
                        stack.push(ceilf(stack.pop()));
                        break;
                    case nnil_cos:
                        stack.push(cosf(stack.pop()));
                        break;
                    case nnil_exp:
                        stack.push(expf(stack.pop()));
                        break;
                    case nnil_floor:
                        stack.push(floorf(stack.pop()));
                        break;
                    case nnil_log:
                        stack.push(logf(stack.pop()));
                        break;
                    case nnil_neg:
                        stack.push(-stack.pop());
                        break;
                    case nnil_rsqrt:
                        stack.push(1.f / sqrtf(stack.pop()));
                        break;
                    case nnil_sin:
                        stack.push(sinf(stack.pop()));
                        break;
                    case nnil_square:
                    {
                        auto v = stack.pop();
                        stack.push(v * v);
                        break;
                    }
                    case nnil_add:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(a + b);
                        break;
                    }
                    case nnil_sub:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(a - b);
                        break;
                    }
                    case nnil_mul:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(a * b);
                        break;
                    }
                    case nnil_div:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(a / b);
                        break;
                    }
                    case nnil_min:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(std::min(a, b));
                        break;
                    }
                    case nnil_max:
                    {
                        auto b = stack.pop();
                        auto a = stack.pop();
                        stack.push(std::max(a, b));
                        break;
                    }
                    case nnil_clamp:
                    {
                        auto high = stack.pop();
                        auto low = stack.pop();
                        auto v = stack.pop();
                        stack.push(std::clamp(v, low, high));
                        break;
                    }
                    case nnil_ret:
                        output[i] = stack.pop();
                        ret = true;
                        break;
                    default:
                        NNCASE_THROW(std::runtime_error, "Invalid nnil op");
                        break;
                    }
                }
            }
        }

        inline void table_lookup1d(const uint8_t *CXX_RESTRICT input, uint8_t *CXX_RESTRICT output, size_t size, const uint8_t *CXX_RESTRICT table)
        {
            for (size_t i = 0; i < size; i++)
                output[i] = table[input[i]];
        }
    }
}
}
