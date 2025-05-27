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
#include "../loop.h"
#include "../shape.h"
#include "../tensor_traits.h"
#include <cstddef>
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {

template <class T1, class T2, bool Arch> struct u_pack_policy {
    static constexpr size_t unroll = 4;
};

template <bool Arch, Scalar TIn, Vector TOut> class u_pack {
  public:
    template <Dimension TM, Dimension TN, Dimension TMStrides>
    constexpr void operator()(const TIn *input, const TM &M, const TN &N,
                              const TMStrides &m_strides,
                              TOut *output) noexcept {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                output[j](i) = input[i * m_strides + j];
            }
        }

        const auto out_length = typename TOut::shape_type{}.length();
        if (M < out_length) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = M; i < out_length; i++) {
                    output[j](i) = (TIn)0;
                }
            }
        }
    }
};

template <bool Arch, class TIn, class TOut, class TElem, class TVec>
class u_pack2d {
  public:
    template <FixedDimensions TAxes>
    constexpr void operator()(const TIn &input, const TAxes &axes,
                              TOut &output) noexcept {
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto lanes = TVec::shape();
        auto out_shape = output.shape();
        const auto domain = out_shape.concat(lanes);
        apply(domain, [&](auto index) {
            const auto out_index = index.template slice<0, out_rank>();
            const auto in_index_template = index.template slice<0, in_rank>();
            const auto elem_index = index.template slice<out_rank>();

            bool skip = false;
            const auto in_index =
                axes.aggregate(in_index_template, [&](const auto &cnt_in_index,
                                                      auto axis, auto i) {
                    const auto in_dim =
                        cnt_in_index[axis] * lanes[i] + index[out_rank + i];
                    if (in_dim >= input.shape()[axis]) {
                        skip = true;
                    }
                    return cnt_in_index.template replace_at<axis>(in_dim);
                });
            output(out_index)(elem_index) = skip ? (TElem)0 : input(in_index);
        });
    }
};
} // namespace ukernels

template <Scalar TIn, Dimension TM, Dimension TN, Dimension TMStrides,
          Vector TOut>
constexpr void u_pack(const TIn *input, const TM &M, const TN &N,
                      const TMStrides &m_strides, TOut *output) noexcept {
    ukernels::u_pack<true, std::decay_t<TIn>, std::decay_t<TOut>> impl;
    impl(input, M, N, m_strides, output);
}

template <class TIn, FixedDimensions TAxes, class TOut>
constexpr void u_pack2d(const TIn &input, const TAxes &axes,
                        TOut &output) noexcept {
    using TElem = typename TIn::element_type;
    using TVec = typename std::decay_t<TOut>::element_type;
    ukernels::u_pack2d<true, TIn, TOut, TElem, TVec> impl;
    impl(input, axes, output);
}
} // namespace nncase::ntt
