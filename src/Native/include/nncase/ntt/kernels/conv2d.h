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
#include "../apply.h"
#include "../loop.h"
#include "../utility.h"

namespace nncase::ntt {
namespace conv_detail {
constexpr inline size_t get_windowed_output_size(size_t size, size_t filter,
                                                 size_t stride, size_t dilation,
                                                 size_t pad_before,
                                                 size_t pad_after) noexcept {
    auto effective_filter_size = (filter - 1) * dilation + 1;
    return (size_t)((int32_t)size + pad_before + pad_after -
                    effective_filter_size + stride) /
           stride;
}

template <IsFixedTensor TInput, IsFixedTensor TWeights, IsFixedTensor TBias,
          IsFixedTensor TOutput, IsFixedDims TStride, IsFixedDims TPadding,
          IsFixedDims TDilation>
void impl(const TInput &input, const TWeights &weights, const TBias &bias,
          TOutput &&output, [[maybe_unused]] const TStride stride,
          [[maybe_unused]] const TPadding padding,
          [[maybe_unused]] const TDilation dilation,
          const size_t groups) noexcept {
    using TElem = TInput::element_type;
    constexpr auto in_shape = typename TInput::shape_type{};
    constexpr int32_t out_channels = TWeights::shape_type::at(0);
    constexpr int32_t filter_h = TWeights::shape_type::at(2);
    constexpr int32_t filter_w = TWeights::shape_type::at(3);
    constexpr int32_t stride_h = TStride::at(0);
    constexpr int32_t stride_w = TStride::at(1);
    constexpr int32_t dilation_h = TDilation::at(0);
    constexpr int32_t dilation_w = TDilation::at(1);
    constexpr int32_t pad_h_before = TPadding::at(0);
    constexpr int32_t pad_h_after = TPadding::at(1);
    constexpr int32_t pad_w_before = TPadding::at(2);
    constexpr int32_t pad_w_after = TPadding::at(3);
    constexpr int32_t out_h = get_windowed_output_size(
        in_shape[2], filter_h, stride_h, dilation_h, pad_h_before, pad_h_after);
    constexpr int32_t out_w = get_windowed_output_size(
        in_shape[3], filter_w, stride_w, dilation_w, pad_w_before, pad_w_after);
    const int32_t g_ic = in_shape[1] / groups;
    const int32_t g_oc = out_channels / groups;

    ranked_shape<4> in_index;
    ranked_shape<4> w_index;
    ranked_shape<1> bias_index;
    ranked_shape<4> out_index;
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = out_index[0] = batch;
        for (size_t og = 0; og < groups; og++) {
            for (size_t oc = 0; oc < g_oc; oc++) {
                out_index[1] = w_index[0] = bias_index[0] = og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++) {
                    out_index[2] = oy;
                    for (size_t ox = 0; ox < out_w; ox++) {
                        out_index[3] = ox;
                        const int32_t in_y_origin =
                            (oy * stride_h) - pad_h_before;
                        const int32_t in_x_origin =
                            (ox * stride_w) - pad_w_before;
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
                        TElem value = bias(bias_index);

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

                                    const TElem in_v = input(in_index);
                                    const TElem w = weights(w_index);

                                    value += in_v * w;
                                }
                            }
                        }

                        output(out_index) = value;
                    }
                }
            }
        }
    }
}
} // namespace conv_detail

template <typename TInput, typename TWeights, typename TBias, typename TOutput,
          IsFixedDims TStride, IsFixedDims TPadding, IsFixedDims TDilation>
void conv2d(const TInput &input, const TWeights &weights, const TBias &bias,
            TOutput &&output, [[maybe_unused]] const TStride stride,
            [[maybe_unused]] const TPadding padding,
            [[maybe_unused]] const TDilation dilation,
            const size_t groups) noexcept {
    conv_detail::impl(input, weights, bias, output, stride, padding, dilation,
                      groups);
}
} // namespace nncase::ntt
