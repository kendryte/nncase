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
#include "../kernel_utils.h"
#include <cmath>
#include <nncase/runtime/nnil.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <xtl/xspan.hpp>
#ifdef __riscv
#include "../riscv/neutral_kernels.h"
#endif

namespace nncase::kernels::neutral
{
template <class TOp, class TShape>
void binary(const float *input_a, const float *input_b, float *output, const TShape &in_a_shape,
    const TShape &in_b_shape, const TShape &out_shape, const value_range<float> &fused_activation, TOp &&op)
{
    // opt. no broadcast
    if (in_a_shape == in_b_shape)
    {
        auto size = kernels::detail::compute_size(in_a_shape);
        for (size_t i = 0; i < size; i++)
        {
            const auto a = input_a[i];
            const auto b = input_b[i];
            output[i] = kernels::detail::apply_activation(op(a, b), fused_activation);
        }
    }
    // fallback
    else
    {
        for (size_t d0 = 0; d0 < out_shape[0]; d0++)
        {
            for (size_t d1 = 0; d1 < out_shape[1]; d1++)
            {
                for (size_t d2 = 0; d2 < out_shape[2]; d2++)
                {
                    for (size_t d3 = 0; d3 < out_shape[3]; d3++)
                    {
                        TShape in_off = { d0, d1, d2, d3 };
                        const auto in_a_off = kernels::detail::get_reduced_offset(in_off, in_a_shape);
                        const auto in_b_off = kernels::detail::get_reduced_offset(in_off, in_b_shape);
                        const auto a = input_a[offset(in_a_shape, in_a_off)];
                        const auto b = input_b[offset(in_b_shape, in_b_off)];
                        output[offset(out_shape, in_off)] = kernels::detail::apply_activation(op(a, b), fused_activation);
                    }
                }
            }
        }
    }
}

template <class TOp, class TShape>
void quantized_binary(const uint8_t *input_a, const uint8_t *input_b, uint8_t *output, const TShape &in_a_shape,
    const TShape &in_b_shape, const TShape &out_shape, int32_t input_a_offset, int32_t input_a_mul, int32_t input_a_shift,
    int32_t input_b_offset, int32_t input_b_mul, int32_t input_b_shift, int32_t output_mul, int32_t output_shift, int32_t output_offset, TOp &&op)
{
    // opt. no broadcast
    if (in_a_shape == in_b_shape)
    {
        auto size = kernels::detail::compute_size(in_a_shape);
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
                        TShape in_off = { d0, d1, d2, d3 };
                        const auto in_a_off = kernels::detail::get_reduced_offset(in_off, in_a_shape);
                        const auto in_b_off = kernels::detail::get_reduced_offset(in_off, in_b_shape);
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

template <class TRange, class TPtrGetter = detail::default_ptr_getter<uint8_t, TRange>>
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

template <class TShape>
void conv2d(const float *input, float *output, const float *weights, const float *bias, const TShape &in_shape,
    int32_t groups, int32_t out_channels, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    const padding &padding_h, const padding &padding_w, const value_range<float> &fused_activation)
{
    const auto out_h = detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = (size_t)out_channels / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        const float *in_batch_p = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

        for (size_t og = 0; og < (size_t)groups; og++)
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

                        for (size_t ic = 0; ic < g_ic; ic++)
                        {
                            const float *in_c_p = in_group_p + (size_t)ic * in_shape[2] * in_shape[3];
                            const float *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                            for (size_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (size_t kx = filter_x_start; kx < filter_x_end; kx++)
                                {
                                    const size_t in_y = in_y_origin + dilation_h * ky;
                                    const size_t in_x = in_x_origin + dilation_w * kx;

                                    const float in_v = in_c_p[in_y * in_shape[3] + in_x];
                                    const float w = w_ic_p[ky * filter_w + kx];

                                    value += in_v * w;
                                }
                            }
                        }

                        *output++ = detail::apply_activation(value, fused_activation);
                    }
                }
            }
        }
    }
}

template <class TShape>
void quantized_conv2d(const uint8_t *input, uint8_t *output, const uint8_t *weights, const int32_t *bias, int32_t input_offset, int32_t filter_offset,
    int32_t output_mul, int32_t output_shift, int32_t output_offset, const TShape &in_shape, int32_t groups, int32_t out_channels,
    int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    const padding &padding_h, const padding &padding_w)
{
    const auto out_h = detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
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

template <class TShape>
void conv2d_transpose(const float *input, float *output, const float *weights, const float *bias, const TShape &in_shape,
    int32_t groups, const TShape &out_shape, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    const padding &padding_h, const padding &padding_w, const value_range<float> &fused_activation)
{
    auto output_size = kernels::detail::compute_size(out_shape);
    std::fill(output, output + output_size, 0.f);
    const auto g_ic = in_shape[1] / groups;
    const auto g_oc = out_shape[1] / groups;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        float *out_batch_p = output + (size_t)batch * out_shape[1] * out_shape[2] * out_shape[3];

        for (size_t g = 0; g < (size_t)groups; g++)
        {
            float *out_group_p = out_batch_p + (size_t)g * g_oc * out_shape[2] * out_shape[3];
            const float *w_group_p = weights + (size_t)g * g_oc * g_ic * filter_h * filter_w;

            for (size_t ic = 0; ic < g_ic; ic++)
            {
                for (size_t iy = 0; iy < in_shape[2]; iy++)
                {
                    for (size_t ix = 0; ix < in_shape[3]; ix++)
                    {
                        const int32_t out_y_origin = (iy * stride_h) - padding_h.before;
                        const int32_t out_x_origin = (ix * stride_w) - padding_w.before;
                        const size_t filter_y_start = (size_t)std::max(0, (-out_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)out_shape[2] - out_y_origin + dilation_h - 1) / dilation_h);
                        const size_t filter_x_start = (size_t)std::max(0, (-out_x_origin + dilation_w - 1) / dilation_w);
                        const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)out_shape[3] - out_x_origin + dilation_w - 1) / dilation_w);
                        const float in_v = *input++;

                        for (size_t oc = 0; oc < g_oc; oc++)
                        {
                            float *out_c_p = out_group_p + (size_t)oc * out_shape[2] * out_shape[3];
                            const float *w_oc_p = w_group_p + (size_t)oc * g_ic * filter_h * filter_w;
                            const float *w_ic_p = w_oc_p + (size_t)ic * filter_h * filter_w;

                            for (size_t ky = filter_y_start; ky < filter_y_end; ky++)
                            {
                                for (size_t kx = filter_x_start; kx < filter_x_end; kx++)
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

    // bias
    auto hw = out_shape[2] * out_shape[3];
    for (size_t i = 0; i < output_size; i++)
        output[i] += bias[i / hw % out_shape[1]];

    if (fused_activation != value_range<float>::full())
    {
        for (size_t i = 0; i < output_size; i++)
            output[i] = detail::apply_activation(output[i], fused_activation);
    }
}

template <class TQ>
void dequantize(const TQ *CXX_RESTRICT input, float *CXX_RESTRICT output, size_t count, const quant_param_t &param)
{
#if __riscv
    riscv_dequantize(input, output, count, param);
#else
    for (size_t i = 0; i < count; i++)
    {
        output[i] = (input[i] - param.zero_point) * param.scale;
    }

#endif
}

inline void matmul(const float *input_a, const float *input_b, float *output, const float *bias, int32_t a_rows, int32_t a_cols, int32_t b_cols, const value_range<float> &fused_activation)
{
    for (int32_t oy = 0; oy < a_rows; oy++)
    {
        for (int32_t ox = 0; ox < b_cols; ox++)
        {
            float value = bias[ox];

            for (int32_t i = 0; i < a_cols; i++)
            {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                value += a * b;
            }

            output[oy * b_cols + ox] = detail::apply_activation(value, fused_activation);
        }
    }
}

inline void quantized_matmul(const uint8_t *input_a, const uint8_t *input_b, uint8_t *output, const int32_t *bias, int32_t a_rows, int32_t a_cols, int32_t b_cols, int32_t input_a_offset, int32_t input_b_offset,
    int32_t output_mul, int32_t output_shift, int32_t output_offset)
{
    for (int32_t oy = 0; oy < a_rows; oy++)
    {
        for (int32_t ox = 0; ox < b_cols; ox++)
        {
            int32_t value = bias[ox];
            for (int32_t i = 0; i < a_cols; i++)
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

template <class T, class TShape, class TPaddings>
void pad(const T *input, T *output, const TShape &in_shape, const TPaddings &paddings, T pad_value)
{
    TShape out_shape = { in_shape[0] + paddings[0].sum(),
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
        auto v = (int32_t)std::nearbyintf(input[i] / param.scale + param.zero_point);
        output[i] = (TQ)std::clamp(v, (int32_t)std::numeric_limits<TQ>::lowest(), (int32_t)std::numeric_limits<TQ>::max());
    }
#endif
}

template <class TReducer, class TShape>
void reduce(const float *input, float *output, float init_value, const TShape &in_shape, const TShape &reduced_shape, TReducer &&reducer)
{
    std::fill(output, output + kernels::detail::compute_size(reduced_shape), init_value);

    for (size_t d0 = 0; d0 < in_shape[0]; d0++)
    {
        for (size_t d1 = 0; d1 < in_shape[1]; d1++)
        {
            for (size_t d2 = 0; d2 < in_shape[2]; d2++)
            {
                for (size_t d3 = 0; d3 < in_shape[3]; d3++)
                {
                    runtime_shape_t in_off = { d0, d1, d2, d3 };
                    auto out_off = kernels::detail::get_reduced_offset(in_off, reduced_shape);
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

template <class TBinaryOp, class TOutputOp, class TShape>
void reduce_window2d(const float *input, float *output, float init_value, const TShape &in_shape, int32_t filter_h, int32_t filter_w,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, const padding &padding_h, const padding &padding_w,
    const value_range<float> &fused_activation, TBinaryOp &&binary_op, TOutputOp &&window_op)
{
    const auto out_h = kernels::detail::get_windowed_output_size(in_shape[2], filter_h, stride_h, dilation_h, padding_h);
    const auto out_w = kernels::detail::get_windowed_output_size(in_shape[3], filter_w, stride_w, dilation_w, padding_w);
    runtime_shape_t out_shape { in_shape[0], in_shape[1], out_h, out_w };

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            for (size_t oy = 0; oy < out_h; oy++)
            {
                for (size_t ox = 0; ox < out_w; ox++)
                {
                    const int32_t in_y_origin = ((int32_t)oy * stride_h) - padding_h.before;
                    const int32_t in_x_origin = ((int32_t)ox * stride_w) - padding_w.before;
                    const size_t filter_y_start = (size_t)std::max(0, (-in_y_origin + dilation_h - 1) / dilation_h);
                    const size_t filter_y_end = (size_t)std::min(filter_h, ((int32_t)in_shape[2] - in_y_origin + dilation_h - 1) / dilation_h);
                    const size_t filter_x_start = (size_t)std::max(0, (-in_x_origin + dilation_w - 1) / dilation_w);
                    const size_t filter_x_end = (size_t)std::min(filter_w, ((int32_t)in_shape[3] - in_x_origin + dilation_w - 1) / dilation_w);
                    float value = init_value;
                    int32_t kernel_count = 0;

                    for (size_t ky = filter_y_start; ky < filter_y_end; ky++)
                    {
                        for (size_t kx = filter_x_start; kx < filter_x_end; kx++)
                        {
                            const size_t in_y = in_y_origin + dilation_h * ky;
                            const size_t in_x = in_x_origin + dilation_w * kx;

                            const float in_v = input[offset(in_shape, { batch, oc, in_y, in_x })];

                            value = binary_op(value, in_v);
                            kernel_count++;
                        }
                    }

                    output[offset(out_shape, { batch, oc, oy, ox })] = kernels::detail::apply_activation(window_op(value, kernel_count), fused_activation);
                }
            }
        }
    }
}

template <class T, class TShape>
void resize_nearest_neighbor(const T *input, T *output, const TShape &in_shape, int32_t out_h, int32_t out_w)
{
    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;

    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        auto in_batch = input + batch * in_shape[1] * in_shape[2] * in_shape[3];

        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            auto in_c = in_batch + oc * in_shape[2] * in_shape[3];

            for (size_t oy = 0; oy < (size_t)out_h; oy++)
            {
                auto in_y = std::min((size_t)floorf(oy * height_scale), in_shape[2] - 1);
                auto in_row = in_c + in_y * in_shape[3];

                for (size_t ox = 0; ox < (size_t)out_w; ox++)
                {
                    auto in_x = std::min((size_t)floorf(ox * width_scale), in_shape[3] - 1);
                    *output++ = in_row[in_x];
                }
            }
        }
    }
}

template <class T, class TShape>
inline void resize_bilinear(const T *input, T *output, const TShape &in_shape, int32_t out_h, int32_t out_w, bool align_corners)
{
    auto height_scale = (float)in_shape[2] / out_h;
    auto width_scale = (float)in_shape[3] / out_w;
    if (align_corners && out_h > 1)
        height_scale = (float)(in_shape[2] - 1) / (out_h - 1);
    if (align_corners && out_w > 1)
        width_scale = (float)(in_shape[3] - 1) / (out_w - 1);

    auto destIdx = 0;
    for (size_t batch = 0; batch < in_shape[0]; batch++)
    {
        auto in_batch = input + (size_t)batch * in_shape[1] * in_shape[2] * in_shape[3];

        for (size_t oc = 0; oc < in_shape[1]; oc++)
        {
            auto in_c = in_batch + (size_t)oc * in_shape[2] * in_shape[3];

            for (size_t oy = 0; oy < (size_t)out_h; oy++)
            {
                auto in_y = oy * height_scale;
                auto in_y0 = (size_t)floorf(in_y);
                auto in_y1 = std::min(in_y0 + 1, in_shape[2] - 1);

                for (size_t ox = 0; ox < (size_t)out_w; ox++)
                {
                    auto in_x = ox * width_scale;
                    auto in_x0 = (size_t)floorf(in_x);
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
    for (int32_t batch = 0; batch < outer_size; batch++)
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

template <class T, class TShape>
void transpose(const T *CXX_RESTRICT input, T *CXX_RESTRICT output, const TShape &in_shape, const TShape &in_strides, const TShape &out_strides, const TShape &perm)
{
    runtime_shape_t out_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++)
        out_shape[i] = in_shape[perm[i]];

    runtime_shape_t i(4), o(4);
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
                    output[offset(out_strides, o)] = input[offset(in_strides, i)];
                }
            }
        }
    }
}

template <class T, class TShape>
void strided_slice(const T *CXX_RESTRICT input, T *CXX_RESTRICT output, const TShape &in_shape, const TShape &begin, const TShape &end, const TShape &strides)
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

inline void nnil_unary_method(const float *input, float *output, size_t count, gsl::span<const gsl::byte> body)
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
            case nnil_acos:
                stack.push(acosf(stack.pop()));
                break;
            case nnil_asin:
                stack.push(asin(stack.pop()));
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
            case nnil_sign:
            {
                auto val = stack.pop();
                stack.push((0 < val) - (val < 0));
                break;
            }
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
            case nnil_pow:
            {
                auto b = stack.pop();
                auto a = stack.pop();
                stack.push(std::pow(a, b));
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
                throw std::runtime_error("Invalid nnil op");
            }
        }
    }
}

inline void table_lookup1d(const uint8_t *CXX_RESTRICT input, uint8_t *CXX_RESTRICT output, size_t size, const uint8_t *CXX_RESTRICT table)
{
    for (size_t i = 0; i < size; i++)
        output[i] = table[input[i]];
}

template <class T, class TShape>
void gru(const T *CXX_RESTRICT input, const T *CXX_RESTRICT w, const T *CXX_RESTRICT r, const T *CXX_RESTRICT b, T *CXX_RESTRICT initial_h, T *CXX_RESTRICT output, T *CXX_RESTRICT output_h,
    const runtime_shape_t &input_shape, const runtime_shape_t &w_shape, int mode)
{
    const int seq_length = input_shape[0];
    const int batch_size = input_shape[1];
    const int input_size = input_shape[2];
    const int num_direction = w_shape[0];
    const int hidden_size = w_shape[1] / 3;

    auto sigmoid = [&](float x) {
        return 1 / (1 + std::exp(-x));
    };
    auto tanh = [&](float x) {
        return std::tanh(x);
    };
    runtime_shape_t out_shape { (size_t)seq_length, (size_t)num_direction, (size_t)batch_size, (size_t)hidden_size };

    auto x_gate_size = batch_size * input_size;
    auto w_gate_size = 3 * hidden_size * input_size;
    auto h_t_size = batch_size * hidden_size;
    auto r_gate_size = 3 * hidden_size * hidden_size;

    auto tmp_a = std::vector<float>(batch_size * hidden_size, 0.f);
    auto tmp_b = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_z = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_r = std::vector<float>(batch_size * hidden_size, 0.f);
    auto gate_h = std::vector<float>(batch_size * hidden_size, 0.f);

    std::vector<int> seq_len_loop;
    for (int l = 0; l < seq_length; l++)
        seq_len_loop.push_back(l);
    if (mode == lstm_direction::kReverse)
        std::reverse(seq_len_loop.begin(), seq_len_loop.end());
    auto x_i = input;
    auto h_t = initial_h;
    auto w_i = w;
    auto r_i = r;
    auto b_i = b;
    for (int d = 0; d < num_direction; d++)
    {
        h_t = initial_h + d * h_t_size;
        w_i = w + d * w_gate_size;
        r_i = r + d * r_gate_size;
        b_i = b + d * 6 * hidden_size;
        if (d == 1)
            std::reverse(seq_len_loop.begin(), seq_len_loop.end());
        for (auto i : seq_len_loop)
        {
            x_i = input + i * x_gate_size;
            // clean gate_z gate_r gate_h
            std::fill(gate_z.begin(), gate_z.end(), 0.f);
            std::fill(gate_r.begin(), gate_r.end(), 0.f);
            std::fill(gate_h.begin(), gate_h.end(), 0.f);

            // clean tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_z = x_i * w_i_z + b_w_z + h_t *r_i_z + b_r_z
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[hs];
                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        tmp_b[bs * hidden_size + hs] += h_t[bs * hidden_size + rs] * r_i[hs * hidden_size + rs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[3 * hidden_size + hs];
                    gate_z[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                }
            }
            // gate_z = sigmoid(gate_z);
            std::transform(gate_z.begin(), gate_z.end(), gate_z.begin(), sigmoid);

            // clear tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_r = x_i * w_i_r + b_w_r + h_t *r_i_r + b_r_r
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[hidden_size * input_size + hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[hidden_size + hs];
                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        tmp_b[bs * hidden_size + hs] += h_t[bs * hidden_size + rs] * r_i[hidden_size * hidden_size + hs * hidden_size + rs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[4 * hidden_size + hs];
                    gate_r[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                }
            }
            // gate_r = sigmoid(gate_r);
            std::transform(gate_r.begin(), gate_r.end(), gate_r.begin(), sigmoid);

            // clear tmp_a tmp_b
            std::fill(tmp_a.begin(), tmp_a.end(), 0.f);
            std::fill(tmp_b.begin(), tmp_b.end(), 0.f);
            // gate_h = x_i * w_i_h + b_w_h + gate_r·h_t *r_i_h + b_r_h
            for (int bs = 0; bs < batch_size; bs++)
            {
                for (int hs = 0; hs < hidden_size; hs++)
                {
                    for (int is = 0; is < input_size; is++)
                    {
                        tmp_a[bs * hidden_size + hs] += x_i[bs * input_size + is] * w_i[2 * hidden_size * input_size + hs * input_size + is];
                    }
                    tmp_a[bs * hidden_size + hs] += b_i[2 * hidden_size + hs];
                    for (int rs = 0; rs < hidden_size; rs++)
                    {
                        // if not linear
                        tmp_b[bs * hidden_size + hs] += gate_r[bs * hidden_size + rs] * h_t[bs * hidden_size + rs] * r_i[2 * hidden_size * hidden_size + hs * hidden_size + rs];
                        // if linear
                        // tmp_b[bs * batch_size + hs] +=  h_t[bs * batch_size + rs] * r_i[hidden_size * hidden_size + hs * hidden_size + rs] + b_i[5 * hidden_size + hs];
                    }
                    tmp_b[bs * hidden_size + hs] += b_i[5 * hidden_size + hs];

                    // if not linear
                    gate_h[bs * hidden_size + hs] = tmp_a[bs * hidden_size + hs] + tmp_b[bs * hidden_size + hs];
                    // if linear
                    // gate_h[bs * batch_size + hs] = tmp_a[bs * batch_size + hs] + gate_r[bs * batch_size + rs] * tmp_b[bs * batch_size + hs];
                }
            }
            // gate_h = tanh(gate_h);
            std::transform(gate_h.begin(), gate_h.end(), gate_h.begin(), tanh);

            for (int k = 0; k < batch_size * hidden_size; k++)
            {
                h_t[k] = (1 - gate_z[k]) * gate_h[k] + gate_z[k] * h_t[k];
                // *output++ = h_t[k];
                output[i * (num_direction * batch_size * hidden_size) + d * (batch_size * hidden_size) + k] = h_t[k];
            }
        }
        // if (mode == lstm_direction::kReverse || d == 1)
        //     h_t.reverse();
        for (int k = 0; k < batch_size * hidden_size; k++)
        {
            output_h[d * (batch_size * hidden_size) + k] = h_t[k];
        }
    }
}

template <class T, class TShape>
void tflite_detection_postprocess(const T *CXX_RESTRICT boxes, const T *CXX_RESTRICT scores, const T *CXX_RESTRICT anchors, T *CXX_RESTRICT output_locations, T *CXX_RESTRICT output_classes, T *CXX_RESTRICT output_scores, T *CXX_RESTRICT output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, NNCASE_UNUSED const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale)
{
    struct CenterSizeEncoding
    {
        float y;
        float x;
        float h;
        float w;
    };
    struct BoxCornerEncoding
    {
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    };
    struct BoxInfo
    {
        int index;
        float score;
    };

    auto compute_iou = [&](const std::vector<BoxCornerEncoding> &box, const int &i, const int &j) {
        auto &box_i = box[i];
        auto &box_j = box[j];
        const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
        const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
        if (area_i <= 0 || area_j <= 0)
            return 0.f;
        const float intersection_y_min = std::max<float>(box_i.ymin, box_j.ymin);
        const float intersection_x_min = std::max<float>(box_i.xmin, box_j.xmin);
        const float intersection_y_max = std::min<float>(box_i.ymax, box_j.ymax);
        const float intersection_x_max = std::min<float>(box_i.xmax, box_j.xmax);
        const float intersection_area = std::max<float>(intersection_y_max - intersection_y_min, 0.0) * std::max<float>(intersection_x_max - intersection_x_min, 0.0);
        return intersection_area / (area_i + area_j - intersection_area);
    };

    const auto num_boxes = (int)anchors_shape[0];
    const auto num_classes_with_background = (int)scores_shape[2]; // num_classes + background
    const auto num_detections_per_class = std::min(detections_per_class, max_detections);
    int label_offset = num_classes_with_background - num_classes;
    // DecodeCenterSizeBoxes： get decoded_boxes
    std::vector<BoxCornerEncoding> decoded_boxes(boxes_shape[1]);
    {

        CenterSizeEncoding box_center_size;
        CenterSizeEncoding scale_values { y_scale, x_scale, h_scale, w_scale };
        CenterSizeEncoding anchor;

        for (int index = 0; index < num_boxes; index++)
        {
            const auto box_encoding_index = index * boxes_shape[2];
            box_center_size = *reinterpret_cast<const CenterSizeEncoding *>(boxes + box_encoding_index);
            anchor = *reinterpret_cast<const CenterSizeEncoding *>(anchors + box_encoding_index);

            auto y_center = static_cast<float>(static_cast<double>(box_center_size.y) / static_cast<double>(scale_values.y) * static_cast<double>(anchor.h) + static_cast<double>(anchor.y));
            auto x_center = static_cast<float>(static_cast<double>(box_center_size.x) / static_cast<double>(scale_values.x) * static_cast<double>(anchor.w) + static_cast<double>(anchor.x));
            auto half_h = static_cast<float>(0.5 * (std::exp(static_cast<double>(box_center_size.h) / static_cast<double>(scale_values.h))) * static_cast<double>(anchor.h));
            auto half_w = static_cast<float>(0.5 * (std::exp(static_cast<double>(box_center_size.w) / static_cast<double>(scale_values.w))) * static_cast<double>(anchor.w));
            decoded_boxes[index].ymin = y_center - half_h;
            decoded_boxes[index].xmin = x_center - half_w;
            decoded_boxes[index].ymax = y_center + half_h;
            decoded_boxes[index].xmax = x_center + half_w;
        }
    }
    // NMS MultiClass
    {
        if (use_regular_non_max_suppression)
        {
            // NMS Regular
            int sorted_indices_size = 0;
            std::vector<BoxInfo> box_info_after_regular_nms(max_detections + num_detections_per_class);
            std::vector<int> num_selected(num_classes);

            // compute nms
            std::vector<float> class_scores(num_boxes);
            std::vector<int> selected;
            selected.reserve(num_detections_per_class);

            for (auto col = 0; col < num_classes - 1; col++)
            {
                const float *scores_base = scores + col + label_offset;
                for (int row = 0; row < num_boxes; row++)
                {
                    // Get scores of boxes corresponding to all anchors for single class
                    class_scores[row] = *scores_base;
                    scores_base += num_classes_with_background;
                }
                // Perform non-maximal suppression on single class
                selected.clear();

                // NMS SingleClass
                {
                    std::vector<int> keep_indices;
                    std::vector<float> keep_scores;
                    // select detection box score above score threshold
                    {
                        for (size_t i = 0; i < class_scores.size(); i++)
                        {
                            if (class_scores[i] >= nms_score_threshold)
                            {
                                keep_scores.emplace_back(class_scores[i]);
                                keep_indices.emplace_back(i);
                            }
                        }
                    }

                    int num_scores_kept = (int)keep_scores.size();
                    std::vector<int> sorted_indices;
                    sorted_indices.resize(num_scores_kept);
                    // DecreasingArgSort
                    {
                        std::iota(sorted_indices.begin(), sorted_indices.begin() + num_scores_kept, 0);
                        std::stable_sort(
                            sorted_indices.begin(), sorted_indices.begin() + num_scores_kept,
                            [&keep_scores](const int i, const int j) { return keep_scores[i] > keep_scores[j]; });
                    }

                    const int output_size = std::min(num_scores_kept, max_detections);
                    selected.clear();
                    int num_active_candidate = num_scores_kept;
                    std::vector<uint8_t> active_box_candidate(num_scores_kept, 1);
                    for (int i = 0; i < num_scores_kept; ++i)
                    {
                        if (num_active_candidate == 0 || (int)selected.size() >= output_size)
                            break;
                        if (active_box_candidate[i] == 1)
                        {
                            selected.push_back(keep_indices[sorted_indices[i]]);
                            active_box_candidate[i] = 0;
                            num_active_candidate--;
                        }
                        else
                        {
                            continue;
                        }
                        for (int j = i + 1; j < num_scores_kept; ++j)
                        {
                            if (active_box_candidate[j] == 1)
                            {

                                float iou = compute_iou(
                                    decoded_boxes, keep_indices[sorted_indices[i]],
                                    keep_indices[sorted_indices[j]]);

                                if (iou > nms_iou_threshold)
                                {
                                    active_box_candidate[j] = 0;
                                    num_active_candidate--;
                                }
                            }
                        }
                    }
                }
                // end NMS SingleClass

                if (selected.empty())
                {
                    continue;
                }
                for (size_t i = 0; i < selected.size(); ++i)
                {
                    box_info_after_regular_nms[sorted_indices_size + i].score = class_scores[selected[i]];
                    box_info_after_regular_nms[sorted_indices_size + i].index = (selected[i] * num_classes_with_background + col + label_offset);
                }

                // In-place merge the original boxes and new selected boxes which are both
                // sorted by scores.
                std::inplace_merge(box_info_after_regular_nms.begin(), box_info_after_regular_nms.begin() + sorted_indices_size,
                    box_info_after_regular_nms.begin() + sorted_indices_size + selected.size(),
                    [](const BoxInfo &a, const BoxInfo &b) { return a.score >= b.score; });

                sorted_indices_size = std::min(sorted_indices_size + static_cast<int>(selected.size()), max_detections);
            }
            // end compute nms result

            // Allocate output tensors
            for (int output_box_index = 0; output_box_index < max_detections; output_box_index++)
            {
                if (output_box_index < sorted_indices_size)
                {
                    const int anchor_index = floor(
                        box_info_after_regular_nms[output_box_index].index / num_classes_with_background);
                    const int class_index = box_info_after_regular_nms[output_box_index].index - anchor_index * num_classes_with_background - label_offset;
                    const float selected_score = box_info_after_regular_nms[output_box_index].score;
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[output_box_index] = decoded_boxes[anchor_index];
                    // detection_classes
                    output_classes[output_box_index] = class_index;
                    // detection_scores
                    output_scores[output_box_index] = selected_score;
                }
                else
                {
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[output_box_index] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    // detection_classes
                    output_classes[output_box_index] = 0.0f;
                    // detection_scores
                    output_scores[output_box_index] = 0.0f;
                }
            }
            output_num_detections[0] = sorted_indices_size;
            box_info_after_regular_nms.clear();
        }
        else
        {
            // Fast NMS

            const int max_categories_per_anchor = max_classes_per_detection;
            const int num_categories_per_anchor = std::min(max_categories_per_anchor, num_classes);

            std::vector<float> max_scores;
            max_scores.resize(num_boxes);
            std::vector<int> sorted_class_indices;
            sorted_class_indices.resize(num_boxes * num_categories_per_anchor);

            for (int row = 0; row < num_boxes; row++)
            {
                const float *box_scores = scores + row * num_classes_with_background + label_offset;
                int *class_indices = sorted_class_indices.data() + row * num_categories_per_anchor;

                // DecreasingPartialArgSort
                if (num_categories_per_anchor == 1)
                {
                    auto arg_max_vector = [&](const T *input_data, int size) {
                        T max_value = input_data[0];
                        int max_index = 0;
                        for (int i = 1; i < size; ++i)
                        {
                            // const T curr_value = input_data[i];
                            if (input_data[i] > max_value)
                            {
                                max_value = input_data[i];
                                max_index = i;
                            }
                        }
                        return max_index;
                    };
                    class_indices[0] = arg_max_vector(box_scores, num_classes);
                }
                else
                {
                    std::iota(class_indices, class_indices + num_classes, 0);
                    std::partial_sort(
                        class_indices, class_indices + num_categories_per_anchor, class_indices + num_classes,
                        [&box_scores](const int i, const int j) { return box_scores[i] > box_scores[j]; });
                }
                // end DecreasingPartialArgSort

                max_scores[row] = box_scores[class_indices[0]];
            }
            std::vector<int> selected;
            // NMS SingleClass
            {
                std::vector<int> keep_indices;
                std::vector<float> keep_scores;
                // select detection box score above score threshold
                {
                    for (size_t i = 0; i < max_scores.size(); i++)
                    {
                        if (max_scores[i] >= nms_score_threshold)
                        {
                            keep_scores.emplace_back(max_scores[i]);
                            keep_indices.emplace_back(i);
                        }
                    }
                }

                int num_scores_kept = (int)keep_scores.size();
                std::vector<int> sorted_indices;
                sorted_indices.resize(num_scores_kept);
                // DecreasingArgSort
                {
                    std::iota(sorted_indices.begin(), sorted_indices.begin() + num_scores_kept, 0);
                    std::stable_sort(
                        sorted_indices.begin(), sorted_indices.begin() + num_scores_kept,
                        [&keep_scores](const int i, const int j) { return keep_scores[i] > keep_scores[j]; });
                }
                const int output_size = std::min(num_scores_kept, max_detections);
                selected.clear();
                int num_active_candidate = num_scores_kept;
                std::vector<uint8_t> active_box_candidate(num_scores_kept, 1);
                for (int i = 0; i < num_scores_kept; ++i)
                {
                    if (num_active_candidate == 0 || (int)selected.size() >= output_size)
                        break;
                    if (active_box_candidate[i] == 1)
                    {
                        selected.push_back(keep_indices[sorted_indices[i]]);
                        active_box_candidate[i] = 0;
                        num_active_candidate--;
                    }
                    else
                    {
                        continue;
                    }
                    for (int j = i + 1; j < num_scores_kept; ++j)
                    {
                        if (active_box_candidate[j] == 1)
                        {

                            float iou = compute_iou(
                                decoded_boxes, keep_indices[sorted_indices[i]],
                                keep_indices[sorted_indices[j]]);
                            if (iou > nms_iou_threshold)
                            {
                                active_box_candidate[j] = 0;
                                num_active_candidate--;
                            }
                        }
                    }
                }
            }
            // end NMS SingleClass

            // Allocate output tensors
            int output_box_index = 0;
            for (const auto &selected_index : selected)
            {
                const float *box_scores = scores + selected_index * num_classes_with_background + label_offset;
                const int *class_indices = sorted_class_indices.data() + selected_index * num_categories_per_anchor;

                for (int col = 0; col < num_categories_per_anchor; ++col)
                {
                    int box_offset = max_categories_per_anchor * output_box_index + col;
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[box_offset] = decoded_boxes[selected_index];
                    // detection_classes
                    output_classes[box_offset] = class_indices[col];
                    // detection_scores
                    output_scores[box_offset] = box_scores[class_indices[col]];
                }
                output_box_index++;
            }
            output_num_detections[0] = output_box_index;
        }
    }
}

inline void layernorm(const float *input, float *output, float *scale, float *bias, runtime_shape_t in_shape, int32_t axis, float epsilon)
{
    auto outer_size = 1;
    auto inner_size = 1;
    for (auto i = 0; i < axis; i++)
        outer_size *= in_shape[i];
    for (auto i = axis; i < in_shape.size(); i++)
        inner_size *= in_shape[i];

    for (int32_t batch = 0; batch < outer_size; batch++)
    {
        auto src = input + batch * inner_size;
        auto dest = output + batch * inner_size;

        float mean1 = 0.f;
        for (size_t i = 0; i < inner_size; i++)
            mean1 += src[i] / inner_size;

        std::vector<float> sub(inner_size, 0.f);
        for (size_t i = 0; i < inner_size; i++)
            sub[i] = src[i] - mean1;

        std::vector<float> pow(inner_size, 0.f);
        for (size_t i = 0; i < inner_size; i++)
            pow[i] = sub[i] * sub[i];

        float mean2 = 0.f;
        for (size_t i = 0; i < inner_size; i++)
            mean2 += pow[i] / inner_size;

        float add = mean2 + epsilon;
        float sqrt = std::sqrt(add);

        std::vector<float> div(inner_size, 0.f);
        for (size_t i = 0; i < inner_size; i++)
            div[i] = sub[i] / sqrt;

        for (size_t i = 0; i < inner_size; i++)
            dest[i] = div[i] * scale[i] + bias[i];
    }
}

template <class T>
void compress(const T *input, const uint8_t *condition, T *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis)
{
    if (axis == (int)input_shape.size())
    {
        for (auto i = 0; i < (int)condition_shape[0]; i++)
        {
            if ((float)*(condition + i) == 0)
            {
                continue;
            }
            *output++ = *(input + i);
        }
    }
    else
    {
        int select_slice = 1;
        for (auto i = axis + 1; i < (int)input_shape.size(); i++)
        {
            select_slice *= input_shape[i];
        }
        for (auto j = 0; j < (int)kernels::detail::compute_size(input_shape); j++)
        {
            auto i = j % (select_slice * input_shape[axis]);
            auto cond_index = i / select_slice;
            if (select_slice != 1 && (cond_index >= condition_shape[0] || condition[cond_index] == 0))
                continue;
            if (select_slice == 1 && (i % input_shape[axis] >= condition_shape[0] || condition[cond_index % input_shape[axis] % condition_shape[0]] == 0))
                continue;
            *output++ = *(input + j);
        }
    }
}
}
