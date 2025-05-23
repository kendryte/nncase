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
#include "../tensor_ops.h"
#include "../utility.h"

namespace nncase::ntt {
namespace im2col_details {
/**
 * @brief
 *  support:
 *   1. no pack
 *   2. packed on the input channel.
 */
template <Tensor TIn, Dimensions TKernel, Dimensions TStrides,
          FixedShape PackedAxes, FixedShape PadedNums, Paddings TPadding,
          Tensor TOut>
    requires(PackedAxes::rank() == 0 ||
             (PackedAxes::rank() == 1 && PackedAxes::at(0) == 1))
void im2col_impl(const TIn &input, [[maybe_unused]] const TKernel &kernel,
                 const TStrides &strides, const TPadding &padding,
                 const PackedAxes packedAxes, const PadedNums padedNums,
                 TOut &output) {
    using TElem = typename TIn::element_type;
    const auto input_shape = input.shape();
    const auto input_strides = input.strides();
    const auto output_shape = output.shape();
    const auto output_strides = output.strides();
    const auto OH = output_shape.template at<2>();
    const auto OW = output_shape.template at<3>();
    const auto [batch, IC, IH, IW] = input_shape;
    const auto pad_h = padding.template at<0>();
    const auto pad_w = padding.template at<1>();
    const auto kernel_h = kernel.template at<0>();
    const auto kernel_w = kernel.template at<1>();
    const auto stride_h = strides.template at<0>();
    const auto stride_w = strides.template at<1>();
    static_assert(contiguous_dims(input_shape, input_strides) == 4 &&
                      contiguous_dims(output_shape, output_strides) == 2,
                  "only support contiguous");

    auto inputSpan = input.elements().begin();
    auto outputSpan = output.elements().begin();
    size_t data_col = 0;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int b = 0; b < batch; b++) {
                    auto data_im =
                        inputSpan + ((b * IC * IH * IW) + (ic * IH * IW));
                    int ih = -pad_h.before + kh;
                    for (int oh = 0; oh < OH; oh++) {
                        int iw = -pad_w.before + kw;
                        for (int ow = 0; ow < OW; ow++) {
                            if (iw >= 0 && iw < IW && ih >= 0 && ih < IH) {
                                outputSpan[data_col++] =
                                    data_im[(ih * IW) + iw];
                            } else {
                                outputSpan[data_col++] = (TElem)0;
                            }

                            iw += stride_w;
                        }

                        ih += stride_h;
                    }
                }
            }
        }
    }
}
} // namespace im2col_details
/**
 * @brief im2col
 *
 * @param image [n,c,h,w]
 * @param output [ic * kh * kw, n * oh * ow]
 */
template <Tensor TIn, Dimensions TKernel, Dimensions TStrides,
          FixedShape PackedAxes, FixedShape PadedNums, Paddings TPadding,
          class TOut>
void im2col(const TIn &input, const TKernel &kernel, const TStrides &strides,
            const TPadding &padding, const PackedAxes &packedAxes,
            const PadedNums &padedNums, TOut &&output) {
    im2col_details::im2col_impl(input, kernel, strides, padding, packedAxes,
                                padedNums, output);
}
} // namespace nncase::ntt
