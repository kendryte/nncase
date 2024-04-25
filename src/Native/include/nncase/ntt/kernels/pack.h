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
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../ukernels.h"

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
template <IsFixedTensor TIn, IsFixedTensor TOut, size_t PackAxis>
class pack_impl<TIn, TOut, PackAxis> {
  public:
    using TVec = typename std::decay_t<TOut>::element_type;

    static inline constexpr size_t VecLen = TVec::shape().length();

    constexpr void operator()(const TIn &input, TOut &output) {
        auto in_p = input.elements().data();
        auto out_p = output.elements().data();
        apply<0>(input, output, in_p, out_p);
    }

  private:
    template <size_t Axis, class TInP, class TOutP>
    constexpr void apply(const TIn &input, TOut &output, TInP in_p,
                         TOutP out_p) {
        if constexpr (Axis < PackAxis) {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply<Axis + 1>(input, output, in_p, out_p);
                in_p += input.strides()[Axis];
                out_p += output.strides()[Axis];
            }
        } else {
            constexpr auto rest_rank = TIn::rank() - Axis - 1;
            constexpr auto conti_dims = std::min(
                {rest_rank, contiguous_dims(TIn::shape(), TIn::strides()),
                 contiguous_dims(TOut::shape(), TOut::strides())});
            constexpr auto m_strides = TIn::strides()[Axis];

            for (size_t i = 0; i < TIn::shape()[Axis] / VecLen; i++) {
                apply_transpose<Axis + 1, conti_dims, VecLen, m_strides>(
                    input, output, in_p, out_p);

                in_p += input.strides()[Axis] * VecLen;
                out_p += output.strides()[Axis];
            }

            // Tail
            constexpr const auto tail_m = TIn::shape()[Axis] % VecLen;
            if constexpr (tail_m) {
                apply_transpose<Axis + 1, conti_dims, tail_m, m_strides>(
                    input, output, in_p, out_p);
            }
        }
    }

    template <size_t Axis, size_t ContiguousDims, size_t M, size_t MStrides,
              class TInP, class TOutP>
    constexpr void apply_transpose(const TIn &input, TOut &output, TInP in_p,
                                   TOutP out_p) {
        if constexpr (Axis + ContiguousDims == TOut::rank()) {
            constexpr auto rest_rank = TOut::rank() - Axis;
            constexpr auto rest_dims =
                slice_fixed_dims<rest_rank, TOut::rank() - rest_rank>(
                    TOut::shape());
            constexpr auto N = rest_dims.length();
            ntt::upack<M, N, MStrides>(in_p, out_p);
        } else {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply_transpose<Axis + 1, ContiguousDims, M, MStrides>(
                    input, output, in_p, out_p);
                in_p += input.strides()[Axis];
                out_p += output.strides()[Axis];
            }
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
