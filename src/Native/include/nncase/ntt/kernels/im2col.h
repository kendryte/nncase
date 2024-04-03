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
#include "../shape_infer/reduce_axis.h"
#include "../tensor_ops.h"
#include "../utility.h"

namespace nncase::ntt {
/**
 * @brief im2col
 *
 * @param image [n,c,h,w]
 * @param output [ic * kh * kw, n * oh * ow]
 */
template <IsFixedTensor TIn, IsFixedDims TKernel, IsFixedDims TStrides,
          IsFixedDims TPadding, IsFixedTensor TOut>
void im2col(const TIn &input, [[maybe_unused]] const TKernel &kernel,
            [[maybe_unused]] const TStrides &strides,
            [[maybe_unused]] const TPadding &padding, TOut &&output) {
    constexpr auto input_shape = typename TIn::shape_type{};
    constexpr auto input_strides = typename TIn::strides_type{};
    constexpr auto output_shape = typename std::decay_t<TOut>::shape_type{};
    constexpr auto output_strides = typename std::decay_t<TOut>::strides_type{};
    // clang-format off
    // (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    constexpr size_t OH = (input_shape.at(2) + TPadding::at(0) + TPadding::at(1) - (1 * (TKernel::at(0) - 1) + 1)) / TStrides::at(0) + 1;
    constexpr size_t OW = (input_shape.at(3) + TPadding::at(2) + TPadding::at(3) - (1 * (TKernel::at(1) - 1) + 1)) / TStrides::at(1) + 1;
    // clang-format on
    constexpr size_t batch = input_shape[0];
    constexpr size_t IC = input_shape[1];
    constexpr size_t IH = input_shape[2];
    constexpr size_t IW = input_shape[3];
    constexpr size_t pad_h_before = TPadding::at(0);
    // constexpr size_t pad_h_after = TPadding::at(1);
    constexpr size_t pad_w_before = TPadding::at(2);
    // constexpr size_t pad_w_after = TPadding::at(3);
    constexpr size_t kernel_h = TKernel::at(0);
    constexpr size_t kernel_w = TKernel::at(1);
    constexpr size_t stride_h = TStrides::at(0);
    constexpr size_t stride_w = TStrides::at(1);
    static_assert(contiguous_dims(input_shape, input_strides) == 4, "");
    static_assert(contiguous_dims(output_shape, output_strides) == 2, "");

    auto inputSpan = input.elements().begin();
    auto outputSpan = output.elements().begin();
    size_t data_col = 0;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int b = 0; b < batch; b++) {
                    auto data_im =
                        inputSpan + ((b * IC * IH * IW) + (ic * IH * IW));
                    int ih = -pad_h_before + kh;
                    for (int oh = 0; oh < OH; oh++) {
                        int iw = -pad_w_before + kw;
                        for (int ow = 0; ow < OW; ow++) {
                            if (iw >= 0 && iw < IW && ih >= 0 && ih < IH) {
                                outputSpan[data_col++] =
                                    data_im[(ih * IW) + iw];
                            } else {
                                outputSpan[data_col++] = 0;
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
} // namespace nncase::ntt