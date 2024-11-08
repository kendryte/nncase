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
#include <cstddef>
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {
template <size_t M, size_t N, size_t MStrides, bool Arch, class TIn, class TOut>
class u_pack {
  public:
    constexpr void operator()(const TIn *input, TOut *output) noexcept {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                output[j](i) = input[i * MStrides + j];
            }
        }

        if constexpr (M < TOut::shape_type::length()) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = M; i < TOut::shape_type::length(); i++) {
                    output[j](i) = (TIn)0;
                }
            }
        }
    }
};

template <class TIn, class TOut, class TElem, class TVec, size_t... Axes>
class u_pack2d {
  public:
    constexpr void operator()(const TIn &input, TOut &output) noexcept {
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto in_rank = TIn::rank();
        constexpr auto out_rank = TOut::rank();
        constexpr auto elem_rank = TVec::rank();
        constexpr auto lanes = TVec::shape();
        auto out_shape = output.shape();
        constexpr auto rank = out_rank + elem_rank;
        ranked_shape<rank> domain{};
        for (size_t i = 0, j = 0; i < rank; i++) {
            if (i < out_rank)
                domain[i] = out_shape[i];
            else
                domain[i] = lanes[j++];
        }

        apply(domain, [&](auto index) {
            auto out_index = slice_index<out_rank>(index);
            auto in_index = slice_index<in_rank>(index);
            auto elem_index = slice_index<elem_rank>(index, out_rank);
            bool skip = false;
            loop<axes.size()>([&](auto i) {
                in_index[axes[i]] =
                    in_index[axes[i]] * lanes[i] + index[out_rank + i];
                if (in_index[axes[i]] >= input.shape()[axes[i]]) {
                    skip = true;
                }
            });
            output(out_index)(elem_index) = skip ? (TElem)0 : input(in_index);
        });
    }
};
} // namespace ukernels

template <size_t M, size_t N, size_t MStrides, class TIn, class TOut>
constexpr void u_pack(const TIn *input, TOut *output) noexcept {
    ukernels::u_pack<M, N, MStrides, true, std::decay_t<TIn>,
                     std::decay_t<TOut>>
        impl;
    impl(input, output);
}

template <class TIn, class TOut, size_t... Axes>
constexpr void u_pack2d(const TIn &input, TOut &output) noexcept {
    using TElem = typename TIn::element_type;
    using TVec = typename std::decay_t<TOut>::element_type;
    ukernels::u_pack2d<TIn, TOut, TElem, TVec, Axes...> impl;
    impl(input, output);
}

} // namespace nncase::ntt
