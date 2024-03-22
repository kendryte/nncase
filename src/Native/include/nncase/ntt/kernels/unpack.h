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
#include "../shape_infer/unpack.h"
#include "unpack_element.h"

namespace nncase::ntt {
namespace detail {

template <class InShape, class InElemShape, class OutShape, class InStrides,
          class OutStrides, size_t... Axes>
class unpack_impl;

template <size_t... InDims, size_t... InElemDims, class OutShape,
          size_t... InStrides, class OutStrides, size_t... Axes>
class unpack_impl<fixed_shape<InDims...>, fixed_shape<InElemDims...>, OutShape,
                  fixed_strides<InStrides...>, OutStrides, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename TIn::element_type;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto rank = TIn::shape_type::rank();
        constexpr auto elem_rank = TVec::shape_type::rank();
        constexpr fixed_shape<InDims..., InElemDims...> domain{};

        apply(domain, [&](auto index) {
            auto in_index = slice_index<rank>(index);
            auto elem_index = slice_index<elem_rank>(index, rank);
            auto out_index = slice_index<rank>(index);
            loop<axes.size()>([&](auto i) {
                out_index[axes[i]] =
                    out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
            });
            if (in_bound(out_index, output.shape())) {
                output(out_index) = input(in_index)(elem_index);
            }
        });
    }
};

template <size_t in_rank, size_t... InElemDims, class OutShape, class InStrides,
          class OutStrides, size_t... Axes>
class unpack_impl<ranked_shape<in_rank>, fixed_shape<InElemDims...>, OutShape,
                  InStrides, OutStrides, Axes...> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &&output) {
        using TVec = typename TIn::element_type;
        constexpr auto axes = std::array<size_t, sizeof...(Axes)>{Axes...};
        constexpr auto rank = in_rank;
        constexpr auto elem_rank = TVec::shape_type::rank();

        auto input_shape = input.shape();
        fixed_shape<InElemDims...> elem_shape{};
        constexpr auto domain_rank = in_rank + elem_rank;
        ranked_shape<domain_rank> domain{};
        for (size_t i = 0, j = 0; i < domain_rank; i++) {
            if (i < in_rank)
                domain[i] = input_shape[i];
            else
                domain[i] = elem_shape[j++];
        }

        apply(domain, [&](auto index) {
            auto in_index = slice_index<rank>(index);
            auto elem_index = slice_index<elem_rank>(index, rank);
            auto out_index = slice_index<rank>(index);
            loop<axes.size()>([&](auto i) {
                out_index[axes[i]] =
                    out_index[axes[i]] * TVec::shape()[i] + index[rank + i];
            });
            if (in_bound(out_index, output.shape())) {
                output(out_index) = input(in_index)(elem_index);
            }
        });
    }
};

} // namespace detail

template <size_t... Axes, class TIn, class TOut>
void unpack(const TIn &input, TOut &&output) noexcept {
    detail::unpack_impl<
        typename TIn::shape_type, typename TIn::element_type::shape_type,
        typename std::decay_t<TOut>::shape_type, typename TIn::strides_type,
        typename std::decay_t<TOut>::strides_type, Axes...>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt
