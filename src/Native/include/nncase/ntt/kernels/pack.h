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

template <size_t OutRank, size_t InRank, size_t... Ints>
constexpr ranked_shape<OutRank>
slice_index(const ranked_shape<InRank> &index, const size_t offset,
            std::index_sequence<Ints...>) noexcept {
    return ranked_shape<OutRank>{index[offset + Ints]...};
}

template <size_t OutRank, size_t InRank>
constexpr ranked_shape<OutRank> slice_index(const ranked_shape<InRank> &index,
                                            const size_t offset = 0) {
    static_assert(OutRank <= InRank, "the out rank must less then inRank");
    return slice_index<OutRank>(index, offset,
                                std::make_index_sequence<OutRank>{});
}

template <class InShape, class OutShape, class OutElemShape, class InStrides,
          class OutStrides, size_t... Axes>
class pack_impl;

template <size_t... InDims, size_t... OutDims, size_t... OutElemDims,
          size_t... InStrides, size_t... OutStrides, size_t... Axes>
class pack_impl<fixed_shape<InDims...>, fixed_shape<OutDims...>,
                fixed_shape<OutElemDims...>, fixed_strides<InStrides...>,
                fixed_strides<OutStrides...>, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TScalar = typename TIn::element_type;
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
