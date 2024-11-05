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
#include "../tensor_ops.h"
#include "../ukernels.h"
#include "../utility.h"

namespace nncase::ntt {
namespace detail {
template <class Shape, class InStrides, class OutStrides> class cast_impl;

template <size_t... Dims, size_t... InStrides, size_t... OutStrides>
class cast_impl<fixed_shape<Dims...>, fixed_strides<InStrides...>,
                fixed_strides<OutStrides...>> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        constexpr size_t rank = sizeof...(Dims);
        ranked_shape<rank> index{};
        constexpr auto conti_dims =
            std::min(contiguous_dims(fixed_shape<Dims...>{},
                                     fixed_strides<InStrides...>{}),
                     contiguous_dims(fixed_shape<Dims...>{},
                                     fixed_strides<OutStrides...>{}));
        apply<TIn, TOut, 0, rank, conti_dims, Dims...>(index, input, output);
    }

  private:
    template <class TIn, class TOut, size_t Axis, size_t Rank,
              size_t ContiguousDims, size_t... RestDims>
    constexpr void apply(ranked_shape<Rank> &index, const TIn &input,
                         TOut &output) {
        if constexpr (ContiguousDims == sizeof...(RestDims)) {
            constexpr auto inner_size = fixed_shape<RestDims...>::length();
            auto input_p =
                input.elements().data() + linear_offset(index, input.strides());
            auto output_p = output.elements().data() +
                            linear_offset(index, output.strides());
            cast_contiguous<inner_size>(input_p, output_p);
        } else {
            apply_next<TIn, TOut, Axis, Rank, ContiguousDims, RestDims...>(
                index, input, output);
        }
    }

    template <class TIn, class TOut, size_t Axis, size_t Rank,
              size_t ContiguousDims, size_t Dim, size_t... RestDims>
    constexpr void apply_next(ranked_shape<Rank> &index, const TIn &input,
                              TOut &output) {
        for (index[Axis] = 0; index[Axis] < Dim; index[Axis]++) {
            apply<TIn, TOut, Axis + 1, Rank, ContiguousDims, RestDims...>(
                index, input, output);
        }
    }

    template <size_t Extent, class T1, class T2>
    constexpr void cast_contiguous(const T1 *input, T2 *output) {
        ntt::u_cast(input, 1, output, 1, Extent);
    }
};

template <size_t Rank, class InStrides, class OutStrides>
class cast_impl<ranked_shape<Rank>, InStrides, OutStrides> {
  public:
    template <class TIn, class TOut>
    constexpr void operator()(const TIn &input, TOut &output) {
        ranked_shape<Rank> index{};
        auto conti_dims =
            std::min(contiguous_dims(input.shape(), input.strides()),
                     contiguous_dims(input.shape(), output.strides()));
        apply<TIn, TOut, 0>(index, conti_dims, input, output);
    }

  private:
    template <class TIn, class TOut, size_t Axis>
    constexpr void apply(ranked_shape<Rank> &index, size_t conti_dims,
                         const TIn &input, TOut &output) {
        const auto outer_dims = Rank - conti_dims;
        if (Axis >= outer_dims) {
            size_t inner_size = 1;
            for (size_t i = outer_dims; i < input.shape().rank(); i++)
                inner_size *= input.shape()[i];
            auto input_p =
                input.buffer().data() + linear_offset(index, input.strides());
            auto output_p =
                output.buffer().data() + linear_offset(index, output.strides());
            cast_contiguous(input_p, output_p, inner_size);
        } else if constexpr (Axis < Rank - 1) {
            const auto dim = input.shape()[Axis];
            for (index[Axis] = 0; index[Axis] < dim; index[Axis]++) {
                apply<TIn, TOut, Axis + 1>(index, conti_dims, input, output);
            }
        }
    }

    template <class T1, class T2>
    constexpr void cast_contiguous(const T1 *input, T2 *output, size_t extent) {
        ntt::u_cast(input, 1, output, 1, extent);
    }
};
} // namespace detail

template <typename TIn, typename TOut>
void cast(const TIn &input, TOut &&output) noexcept {
    detail::cast_impl<common_shape_t<typename TIn::shape_type,
                                     typename std::decay_t<TOut>::shape_type>,
                      typename TIn::strides_type,
                      typename std::decay_t<TOut>::strides_type>
        impl;
    impl(input, output);
}
} // namespace nncase::ntt