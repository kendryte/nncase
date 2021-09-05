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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/k210/compiler_defs.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/runtime/k210/runtime_types.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_op_utility.h>

BEGIN_NS_NNCASE_KERNELS_K210

namespace detail
{
template <class T>
struct pool_partial_type;

template <>
struct pool_partial_type<uint8_t>
{
    using type = uint32_t;
};

template <>
struct pool_partial_type<float>
{
    using type = float;
};

template <class T>
using pool_partial_type_t = typename pool_partial_type<T>::type;
}

result<void> kpu_upload(const uint8_t *src, uint8_t *dest, const runtime::k210::kpu_shape_t &in_shape, uint32_t dma_ch);

inline result<void> kpu_download(const uint8_t *src, uint8_t *dest, const runtime::k210::kpu_shape_t &in_shape)
{
    using namespace runtime::k210;

    if (in_shape[3] % 64 == 0)
    {
        std::copy(src, src + kernels::detail::compute_size(in_shape), dest);
    }
    else
    {
        auto layout = get_kpu_row_layout(in_shape[3]);
        auto fmap_size = get_kpu_bytes(in_shape[3], in_shape[2], in_shape[1]);

        for (uint32_t batch = 0; batch < in_shape[0]; batch++)
        {
            auto batch_origin = src + (size_t)batch * fmap_size;
            for (uint32_t oc = 0; oc < in_shape[1]; oc++)
            {
                auto channel_origin = batch_origin + (size_t)oc / layout.groups * layout.row_len * in_shape[2] * 64 + (size_t)oc % layout.groups * layout.row_pitch;
                for (uint32_t y = 0; y < in_shape[2]; y++)
                {
                    auto y_origin = channel_origin + (size_t)y * layout.row_len * 64;
                    for (uint32_t x = 0; x < in_shape[3]; x++)
                        *dest++ = y_origin[x];
                }
            }
        }
    }

    return ok();
}

template <bool IsDepthwise, int32_t FilterSize>
void kpu_conv2d(const uint8_t *input, int64_t *workspace, uint8_t *output, const uint8_t *weights, int32_t in_h, int32_t in_w, int32_t in_channels, int32_t out_channels, uint8_t pad_value, int32_t arg_x,
    int32_t shift_x, int32_t arg_w, int32_t shift_w, int64_t arg_add, const runtime::k210::kpu_batchnorm_segment *batchnorm, const runtime::k210::kpu_activation_table_t &activation)
{
    const auto channel_size = size_t(in_h) * in_w;
    // conv
    {
        auto out_it = workspace;
        const auto pad = FilterSize == 1 ? 0 : 1;
        const auto groups = IsDepthwise ? out_channels : 1;
        const auto g_ic = IsDepthwise ? 1 : in_channels / groups;
        const auto g_oc = IsDepthwise ? 1 : out_channels;

        for (int32_t og = 0; og < groups; og++)
        {
            const uint8_t *w_group_p = weights + (size_t)og * g_oc * g_ic * FilterSize * FilterSize;

            for (int32_t oc = 0; oc < g_oc; oc++)
            {
                const uint8_t *w_oc_p = w_group_p + (size_t)oc * g_ic * FilterSize * FilterSize;

                for (int32_t oy = 0; oy < in_h; oy++)
                {
                    for (int32_t ox = 0; ox < in_w; ox++)
                    {
                        const int32_t in_y_origin = oy - pad;
                        const int32_t in_x_origin = ox - pad;
                        int64_t value = 0;
                        int64_t sum_x = 0, sum_w = 0;

                        for (int32_t ic = 0; ic < g_ic; ic++)
                        {
                            const uint8_t *in_c_p = input + ((size_t)og * g_ic + ic) * in_h * in_w;
                            const uint8_t *w_ic_p = w_oc_p + (size_t)ic * FilterSize * FilterSize;

                            for (int32_t ky = 0; ky < FilterSize; ky++)
                            {
                                for (int32_t kx = 0; kx < FilterSize; kx++)
                                {
                                    const int32_t in_y = in_y_origin + ky;
                                    const int32_t in_x = in_x_origin + kx;

                                    uint8_t x;
                                    if (in_x < 0 || in_x >= in_w
                                        || in_y < 0 || in_y >= in_h)
                                        x = pad_value;
                                    else
                                        x = in_c_p[in_y * in_w + in_x];

                                    uint8_t w = w_ic_p[ky * FilterSize + kx];

                                    sum_x += x;
                                    sum_w += w;
                                    value += (int32_t)x * w;
                                }
                            }
                        }

                        auto alu_out = value + (arg_x * sum_x >> shift_x) + (arg_w * sum_w >> shift_w) + arg_add * g_ic;
                        assert(runtime::within_range<36>(alu_out));
                        *out_it++ = alu_out;
                    }
                }
            }
        }
    }

    // bn act
    {
        auto src_it = workspace;
        auto out_it = output;
        for (int32_t oc = 0; oc < out_channels; oc++)
        {
            const auto &bn = batchnorm[oc];
            for (size_t i = 0; i < channel_size; i++)
            {
                auto value = (*src_it++ * bn.mul >> bn.shift) + bn.add;
                assert(runtime::within_range<36>(value));
                auto &seg = *std::find_if(activation.rbegin(), activation.rend(), [value](const runtime::k210::kpu_activation_segment &seg) { return value > seg.start_x; });
                auto act_value = runtime::carry_shift<int64_t, true>((value - seg.start_x) * seg.mul, seg.shift) + seg.add;
                *out_it++ = (uint8_t)kernels::detail::clamp(act_value, int64_t(0), int64_t(255));
            }
        }
    }
}

template <class T>
inline void kpu_pool2d(const T *input, T *output, int32_t in_h, int32_t in_w, int32_t in_channels, runtime::k210::kpu_pool_type_t pool_type)
{
    using namespace runtime::k210;
    using partial_t = detail::pool_partial_type_t<T>;

    const auto filter = get_kpu_filter_size(pool_type);
    const auto stride = get_kpu_filter_stride(pool_type);
    const auto out_h = get_kpu_pool_output_size(in_h, pool_type);
    const auto out_w = get_kpu_pool_output_size(in_w, pool_type);

    for (int32_t oc = 0; oc < in_channels; oc++)
    {
        auto in_c_p = input + (size_t)oc * in_h * in_w;

        for (int32_t oy = 0; oy < out_h; oy++)
        {
            for (int32_t ox = 0; ox < out_w; ox++)
            {
                const int32_t in_y_origin = oy * stride;
                const int32_t in_x_origin = ox * stride;
                partial_t value = 0;

                switch (pool_type)
                {
                case kpu_pool_bypass:
                {
                    const int32_t in_y = in_y_origin;
                    const int32_t in_x = in_x_origin;

                    value = in_c_p[in_y * in_w + in_x];
                    break;
                }
                case kpu_pool_max_2_s2:
                case kpu_pool_max_2_s1:
                case kpu_pool_max_4_s4:
                {
                    value = std::numeric_limits<T>::lowest();
                    for (int32_t ky = 0; ky < filter; ky++)
                    {
                        for (int32_t kx = 0; kx < filter; kx++)
                        {
                            const int32_t in_y = in_y_origin + ky;
                            const int32_t in_x = in_x_origin + kx;
                            partial_t in_v;

                            if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                                in_v = std::numeric_limits<T>::lowest();
                            else
                                in_v = in_c_p[in_y * in_w + in_x];

                            value = std::max(value, in_v);
                        }
                    }

                    break;
                }
                case kpu_pool_mean_2_s2:
                case kpu_pool_mean_2_s1:
                case kpu_pool_mean_4_s4:
                {
                    for (int32_t ky = 0; ky < filter; ky++)
                    {
                        for (int32_t kx = 0; kx < filter; kx++)
                        {
                            const int32_t in_y = kernels::detail::clamp(in_y_origin + ky, 0, in_h - 1);
                            const int32_t in_x = kernels::detail::clamp(in_x_origin + kx, 0, in_w - 1);
                            const T in_v = in_c_p[in_y * in_w + in_x];

                            value += in_v;
                        }
                    }

                    value /= filter * filter;
                    break;
                }
                case kpu_pool_left_top_2_s2:
                case kpu_pool_left_top_4_s4:
                case kpu_pool_right_top_2_s2:
                {
                    auto k_off = get_kpu_select_pool_offset(pool_type);
                    const int32_t in_y = in_y_origin + k_off[0];
                    const int32_t in_x = in_x_origin + k_off[1];
                    partial_t in_v;

                    if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                        in_v = 0;
                    else
                        in_v = in_c_p[in_y * in_w + in_x];

                    value = in_v;
                    break;
                }
                }

                *output++ = (T)value;
            }
        }
    }
}

template <bool IsDepthwise, int32_t FilterSize>
void fake_kpu_conv2d(const float *input, float *output, const float *weights, const float *bias, int32_t in_h, int32_t in_w, int32_t in_channels, int32_t out_channels, value_range<float> fused_activation)
{
    const auto pad = FilterSize == 1 ? 0 : 1;
    const auto groups = IsDepthwise ? out_channels : 1;
    const auto g_ic = IsDepthwise ? 1 : in_channels / groups;
    const auto g_oc = IsDepthwise ? 1 : out_channels;

    for (int32_t og = 0; og < groups; og++)
    {
        const auto *w_group_p = weights + (size_t)og * g_oc * g_ic * FilterSize * FilterSize;

        for (int32_t oc = 0; oc < g_oc; oc++)
        {
            const auto *w_oc_p = w_group_p + (size_t)oc * g_ic * FilterSize * FilterSize;

            for (int32_t oy = 0; oy < in_h; oy++)
            {
                for (int32_t ox = 0; ox < in_w; ox++)
                {
                    const int32_t in_y_origin = oy - pad;
                    const int32_t in_x_origin = ox - pad;
                    const int32_t filter_y_start = std::max(0, -in_y_origin);
                    const int32_t filter_y_end = std::min(FilterSize, in_h - in_y_origin);
                    const int32_t filter_x_start = std::max(0, -in_x_origin);
                    const int32_t filter_x_end = std::min(FilterSize, in_w - in_x_origin);
                    float value = bias[og * g_oc + oc];

                    for (int32_t ic = 0; ic < g_ic; ic++)
                    {
                        const auto *in_c_p = input + ((size_t)og * g_ic + ic) * in_h * in_w;
                        const auto *w_ic_p = w_oc_p + (size_t)ic * FilterSize * FilterSize;

                        for (int32_t ky = filter_y_start; ky < filter_y_end; ky++)
                        {
                            for (int32_t kx = filter_x_start; kx < filter_x_end; kx++)
                            {
                                const int32_t in_y = in_y_origin + ky;
                                const int32_t in_x = in_x_origin + kx;

                                const auto in_v = in_c_p[in_y * in_w + in_x];
                                const auto w = w_ic_p[ky * FilterSize + kx];

                                value += in_v * w;
                            }
                        }
                    }

                    *output++ = kernels::detail::apply_activation(value, fused_activation);
                }
            }
        }
    }
}

END_NS_NNCASE_KERNELS_K210
