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

namespace nncase::ntt {
namespace detail {
template <class Shape, class InStrides, class OutStrides> class unary_impl;

template <size_t... Dims, size_t... InStrides, size_t... OutStrides>
class unary_impl<fixed_shape<Dims...>, fixed_strides<InStrides...>,
                 fixed_strides<OutStrides...>> {
  public:
    template <class Op, class TIn, class TOut>
    constexpr void operator()(Op &op, const TIn &input, TOut &output) {
        constexpr size_t rank = sizeof...(Dims);
        ranked_shape<rank> index{};
        constexpr auto conti_dims =
            std::min(contiguous_dims(fixed_shape<Dims...>{},
                                     fixed_strides<InStrides...>{}),
                     contiguous_dims(fixed_shape<Dims...>{},
                                     fixed_strides<OutStrides...>{}));
        apply<Op, TIn, TOut, 0, rank, conti_dims, Dims...>(op, index, input,
                                                           output);
    }

  private:
    template <class Op, class TIn, class TOut, size_t Axis, size_t Rank,
              size_t ContiguousDims, size_t... RestDims>
    constexpr void apply(Op &op, ranked_shape<Rank> &index, const TIn &input,
                         TOut &output) {
        if constexpr (ContiguousDims == sizeof...(RestDims)) {
            constexpr auto inner_size = fixed_shape<RestDims...>::length();
            auto input_p =
                input.buffer().data() + linear_offset(index, input.strides());
            auto output_p =
                output.buffer().data() + linear_offset(index, output.strides());
            unary_contiguous<inner_size>(op, input_p, output_p);
        } else {
            apply_next<Op, TIn, TOut, Axis, Rank, ContiguousDims, RestDims...>(
                op, index, input, output);
        }
    }

    template <class Op, class TIn, class TOut, size_t Axis, size_t Rank,
              size_t ContiguousDims, size_t Dim, size_t... RestDims>
    constexpr void apply_next(Op &op, ranked_shape<Rank> &index,
                              const TIn &input, TOut &output) {
        for (index[Axis] = 0; index[Axis] < Dim; index[Axis]++) {
            apply<Op, TIn, TOut, Axis + 1, Rank, ContiguousDims, RestDims...>(
                op, index, input, output);
        }
    }

    template <size_t Extent, class T, class Op>
    constexpr void unary_contiguous(Op &&op, const T *input_p, T *output_p) {
        for (size_t i = 0; i < Extent; i++) {
            output_p[i] = op(input_p[i]);
        }
    }
};

template <size_t Rank, class InStrides, class OutStrides>
class unary_impl<ranked_shape<Rank>, InStrides, OutStrides> {
  public:
    template <class Op, class TIn, class TOut>
    constexpr void operator()(Op &op, const TIn &input, TOut &output) {
        ranked_shape<Rank> index{};
        auto conti_dims =
            std::min(contiguous_dims(input.shape(), input.strides()),
                     contiguous_dims(input.shape(), output.strides()));
        apply<Op, TIn, TOut, 0>(op, index, conti_dims, input, output);
    }

  private:
    template <class Op, class TIn, class TOut, size_t Axis>
    constexpr void apply(Op &op, ranked_shape<Rank> &index, size_t conti_dims,
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
            unary_contiguous(op, input_p, output_p, inner_size);
        } else if constexpr (Axis < Rank - 1) {
            const auto dim = input.shape()[Axis];
            for (index[Axis] = 0; index[Axis] < dim; index[Axis]++) {
                apply<Op, TIn, TOut, Axis + 1>(op, index, conti_dims, input,
                                               output);
            }
        }
    }

    template <class T, class Op>
    constexpr void unary_contiguous(Op &&op, const T *input_p, T *output_p,
                                    size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            output_p[i] = op(input_p[i]);
        }
    }
};
} // namespace detail

template <template <class T> class Op, class TIn, class TOut>
void unary(const TIn &input, TOut &&output) {
    Op<typename TIn::element_type> op;
    detail::unary_impl<common_shape_t<typename TIn::shape_type,
                                      typename std::decay_t<TOut>::shape_type>,
                       typename TIn::strides_type,
                       typename std::decay_t<TOut>::strides_type>
        impl;
    impl(op, input, output);
}
} // namespace nncase::ntt
