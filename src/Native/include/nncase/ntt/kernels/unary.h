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
// math ops
namespace mathops {
template <class T> struct abs {
    T operator()(T v) const noexcept { return std::abs(v); }
};

template <class T> struct acos {
    T operator()(T v) const noexcept { return std::acos(v); }
};

template <class T> struct acosh {
    T operator()(T v) const noexcept { return std::acosh(v); }
};

template <class T> struct asin {
    T operator()(T v) const noexcept { return std::asin(v); }
};

template <class T> struct asinh {
    T operator()(T v) const noexcept { return std::asinh(v); }
};

template <class T> struct ceil {
    T operator()(T v) const noexcept { return std::ceil(v); }
};

template <class T> struct cos {
    T operator()(T v) const noexcept { return std::cos(v); }
};

template <class T> struct cosh {
    T operator()(T v) const noexcept { return std::cosh(v); }
};

template <class T> struct exp {
    T operator()(T v) const noexcept { return std::exp(v); }
};

template <class T> struct floor {
    T operator()(T v) const noexcept { return std::floor(v); }
};

template <class T> struct log {
    T operator()(T v) const noexcept { return std::log(v); }
};

template <class T> struct neg {
    T operator()(T v) const noexcept { return -v; }
};

template <class T> struct round {
    T operator()(T v) const noexcept { return std::nearbyint(v); }
};

template <class T> struct rsqrt {
    T operator()(T v) const noexcept { return (T)1 / std::sqrt(v); }
};

template <class T> struct sign {
    T operator()(T v) const noexcept { return std::copysign((T)1, v); }
};

template <class T> struct sin {
    T operator()(T v) const noexcept { return std::sin(v); }
};

template <class T> struct sinh {
    T operator()(T v) const noexcept { return std::sinh(v); }
};

template <class T> struct sqrt {
    T operator()(T v) const noexcept { return std::sqrt(v); }
};

template <class T> struct square {
    T operator()(T v) const noexcept { return v * v; }
};

template <class T> struct tanh {
    T operator()(T v) const noexcept { return std::tanh(v); }
};
} // namespace mathops

template <size_t Extent, class T, class Op>
constexpr void unary(Op &&op, const T *input_p, T *output_p) {
    for (size_t i = 0; i < Extent; i++) {
        output_p[i] = op(input_p[i]);
    }
}

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
            unary<inner_size>(op, input_p, output_p);
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
            apply<Op, TIn, TOut, Axis, Rank, ContiguousDims, RestDims...>(
                op, index, input, output);
        }
    }
};
} // namespace detail

template <template <class T> class Op, class TIn, class TOut>
void unary(const TIn &input, TOut &&output) {
    Op<typename TIn::element_type> op;
    detail::unary_impl<typename TIn::shape_type, typename TIn::strides_type,
                       typename std::decay_t<TOut>::strides_type>
        impl;
    impl(op, input, output);
}
} // namespace nncase::ntt
