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
#include "../shape_infer/pack.h"
#include "pack_element.h"

namespace nncase::ntt {
namespace detail {

template <class InShape, class OutShape, class OutElemShape, class InStrides,
          class OutStrides, size_t... Axes>
class pack_impl;

template <class InShape, size_t... OutDims, size_t... OutElemDims,
          class InStrides, size_t... OutStrides, size_t... Axes>
class pack_impl<InShape, fixed_shape<OutDims...>, fixed_shape<OutElemDims...>,
                InStrides, fixed_strides<OutStrides...>, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename std::decay_t<TOut>::element_type;
        constexpr fixed_shape<OutDims..., OutElemDims...> domain{};
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto out_rank = std::decay_t<TOut>::shape_type::rank();
        constexpr auto in_rank = TIn::shape_type::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr auto lanes = typename TVec::shape_type{};

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
            output(out_index)(elem_index) = skip ? 0 : input(in_index);
        });
    }
};

template <class InShape, size_t out_rank, size_t... OutElemDims,
          class InStrides, class OutStrides, size_t... Axes>
class pack_impl<InShape, ranked_shape<out_rank>, fixed_shape<OutElemDims...>,
                InStrides, OutStrides, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename std::decay_t<TOut>::element_type;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto in_rank = TIn::shape_type::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr auto lanes = typename TVec::shape_type{};

        auto out_shape = output.shape();
        auto OutElemShape = fixed_shape<OutElemDims...>{};
        constexpr auto rank = out_rank + sizeof...(OutElemDims);
        ranked_shape<rank> domain{};
        for (size_t i = 0, j = 0; i < rank; i++) {
            if (i < out_rank)
                domain[i] = out_shape[i];
            else
                domain[i] = OutElemShape[j++];
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
            output(out_index)(elem_index) = skip ? 0 : input(in_index);
        });
    }
};

} // namespace detail

template <size_t... Axes, class TIn, class TOut>
void pack(const TIn &input, TOut &&output) noexcept {
    detail::pack_impl<typename TIn::shape_type,
                      typename std::decay_t<TOut>::shape_type,
                      typename std::decay_t<TOut>::element_type::shape_type,
                      typename TIn::strides_type,
                      typename std::decay_t<TOut>::strides_type, Axes...>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
