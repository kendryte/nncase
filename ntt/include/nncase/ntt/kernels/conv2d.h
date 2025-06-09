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
#include "../padding.h"
#include "../shape_infer/window.h"
#include "../utility.h"

namespace nncase::ntt {
namespace conv_detail {
template <Tensor TInput, Tensor TWeights, Tensor TBias, Tensor TOutput,
          Dimensions TStride, Paddings TPadding, Dimensions TDilation,
          Dimension TGroups>
void impl(const TInput &input, TOutput &output, const TWeights &weights,
          const TBias &bias, const TStride &stride, const TPadding &padding,
          const TDilation &dilation, const TGroups &groups) noexcept {
    using TElem = typename TInput::element_type;
    const auto in_shape = input.shape();
    const auto out_channels = weights.shape.template at<0>();
    const auto filter_h = weights.shape.template at<2>();
    const auto filter_w = weights.shape.template at<3>();
    const auto stride_h = stride.template at<0>();
    const auto stride_w = stride.template at<1>();
    const auto dilation_h = dilation.template at<0>();
    const auto dilation_w = dilation.template at<1>();
    const auto pad_h = padding.template at<0>();
    const auto pad_w = padding.template at<1>();
    const auto out_h = shape_infer::windowed_output_size(
        in_shape.template at<2>(), filter_h, stride_h, dilation_h, pad_h);
    const auto out_w = shape_infer::windowed_output_size(
        in_shape.template at<3>(), filter_w, stride_w, dilation_w, pad_w);
    const auto g_ic = in_shape.template at<1>() / groups;
    const auto g_oc = out_channels / groups;

    dynamic_shape_t<4> in_index;
    dynamic_shape_t<4> w_index;
    dynamic_shape_t<1> bias_index;
    dynamic_shape_t<4> out_index;
    for (size_t batch = 0; batch < in_shape.template at<0>(); batch++) {
        in_index.at<0>() = out_index.at<0>() = batch;
        for (size_t og = 0; og < groups; og++) {
            for (size_t oc = 0; oc < g_oc; oc++) {
                out_index.at<1>() = w_index.at<0>() = bias_index.at<0>() =
                    og * g_oc + oc;
                for (size_t oy = 0; oy < out_h; oy++) {
                    out_index.at<2>() = oy;
                    for (size_t ox = 0; ox < out_w; ox++) {
                        out_index.at<3>() = ox;
                        const int32_t in_y_origin =
                            (oy * stride_h) - pad_h.before;
                        const int32_t in_x_origin =
                            (ox * stride_w) - pad_w.before;
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
                            in_index.at<1>() = og * g_ic + ic;
                            w_index.at<1>() = ic;
                            for (int32_t ky = filter_y_start; ky < filter_y_end;
                                 ky++) {
                                w_index.at<2>() = ky;
                                for (int32_t kx = filter_x_start;
                                     kx < filter_x_end; kx++) {
                                    w_index.at<3>() = kx;
                                    in_index.at<2>() =
                                        in_y_origin + dilation_h * ky;
                                    in_index.at<3>() =
                                        in_x_origin + dilation_w * kx;

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

template <Tensor TInput, Tensor TWeights, Tensor TBias, class TOutput,
          Dimensions TStride, Paddings TPadding, Dimensions TDilation,
          Dimension TGroups>
void conv2d(const TInput &input, TOutput &&output, const TWeights &weights,
            const TBias &bias, const TStride &stride, const TPadding &padding,
            const TDilation &dilation, const TGroups &groups) noexcept {
    conv_detail::impl(input, output, weights, bias, stride, padding, dilation,
                      groups);
}
} // namespace nncase::ntt
