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
#include "../tensor_traits.h"

namespace nncase::ntt {
namespace detail {
template <class TIn, class TOut, size_t... Axes> class pack_impl {
  public:
    using TVec = typename std::decay_t<TOut>::element_type;

    constexpr void operator()(const TIn &input, TOut &output) {
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
            output(out_index)(elem_index) = skip ? 0 : input(in_index);
        });
    }
};

// 1D packing
template <IsFixedTensor TIn, IsFixedTensor TOut, size_t Axis>
class pack_impl<TIn, TOut, Axis> {
  public:
    using TVec = typename std::decay_t<TOut>::element_type;

    constexpr void operator()(const TIn &input, TOut &output) {
        // 1. Last axis, no transpose
        if constexpr (Axis == TIn::rank() - 1) {
            constexpr auto conti_dims =
                std::min({contiguous_dims(TIn::shape(), TIn::strides()),
                          contiguous_dims(TOut::shape(), TOut::strides())});
            auto in_p = input.elements().data();
            auto out_p = output.elements().data();
            apply_no_transpose<0, conti_dims>(input, output, in_p, out_p);
        } else {

        }
    }

  private:
    template <size_t CntAxis, size_t ContiguousDims, class TInP, class TOutP>
    constexpr void apply_no_transpose(const TIn &input, TOut &output, TInP in_p,
                                      TOutP out_p) {
        // 1. In contiguous axes
        if constexpr (CntAxis + ContiguousDims >= TOut::rank()) {
            constexpr auto rest_rank = TIn::rank() - Axis;
            constexpr auto in_rest_dims =
                slice_fixed_dims<rest_rank, TIn::rank() - rest_rank>(
                    TIn::shape());
            load_contiguous(in_p, out_p, in_rest_dims.length());
        } else {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply_no_transpose<CntAxis + 1, ContiguousDims>(input, output,
                                                                in_p, out_p);
                in_p += input.strides()[CntAxis];
                out_p += output.strides()[CntAxis];
            }
        }
    }

    template <class TInElem, class TOutElem>
    void load_contiguous(const TInElem *input, TOutElem *output,
                         size_t extent) {
        constexpr auto vec_len = TVec::shape().length();
        while (extent) {
            const auto len = std::min(extent, vec_len);
            *output++ = ntt::tload<TVec>(input);
            input += len;
            extent -= len;
        }
    }
};
} // namespace detail

template <size_t... Axes, class TIn, class TOut>
void pack(const TIn &input, TOut &&output) noexcept {
    detail::pack_impl<std::decay_t<TIn>, std::decay_t<TOut>, Axes...> impl;
    impl(input, output);
}
} // namespace nncase::ntt
