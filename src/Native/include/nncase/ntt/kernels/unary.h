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

namespace detail {
template <template <class T> class Op, class TA, class TB>
class unary_apply_impl {
  public:
    using T = typename TA::element_type;

    constexpr unary_apply_impl(const TA &input, TB &output)
        : input_(input), output_(output) {}

    constexpr void operator()() {
        apply(std::make_index_sequence<input_.shape().rank()>());
    }

  private:
    template <size_t... Axes>
    constexpr void apply(std::index_sequence<Axes...>) {
        constexpr size_t conti_dims =
            std::min(contiguous_dims(input_.shape(), input_.strides()),
                     contiguous_dims(output_.shape(), output_.strides()));
        ranked_shape<input_.shape().rank()> index{};
        apply<0, sizeof...(Axes), conti_dims, input_.shape().at(Axes)...>(
            index);
    }

    template <size_t Axis, size_t Rank, size_t ContiguousDims, size_t... Dims>
    constexpr void apply(ranked_shape<Rank> &index) {
        if constexpr (ContiguousDims == sizeof...(Dims)) {
            constexpr auto inner_size = (Dims * ... * 1);
            auto input_p =
                input_.buffer().data() + linear_offset(index, input_.strides());
            auto output_p = output_.buffer().data() +
                            linear_offset(index, output_.strides());
            apply_contiguous<inner_size>(input_p, output_p);
        } else {
            apply_next<Axis + 1, Rank, ContiguousDims, Dims...>(index);
        }
    }

    template <size_t Axis, size_t Rank, size_t ContiguousDims, size_t Dim,
              size_t... Dims>
    constexpr void apply_next(ranked_shape<Rank> &index) {
        for (index[Axis] = 0; index[Axis] < Dim; index[Axis]++) {
            apply<Axis, Rank, ContiguousDims, Dims...>(index);
        }
    }

    template <size_t InnerSize>
    constexpr void apply_contiguous(const T *input_p, T *output_p) {
        for (size_t i = 0; i < InnerSize; i++) {
            output_p[i] = op_(input_p[i]);
        }
    }

  private:
    const TA &input_;
    TB &output_;
    Op<T> op_;
};
} // namespace detail

template <template <class T> class Op, class TA, class TB>
void unary(const TA &input, TB &&output) {
    detail::unary_apply_impl<Op, TA, TB> apply(input, output);
    apply();
}
} // namespace nncase::ntt
