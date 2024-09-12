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
#include "../primitive_ops.h"
#include "../profiler.h"
#include "../shape_infer/reduce.h"
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"
#include <limits>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
enum class reduce_op {
    mean,
    min,
    max,
    sum,
    prod,
};

namespace detail {
template <reduce_op Op> struct reduce_to_binary_type;

template <> struct reduce_to_binary_type<reduce_op::mean> {
    template <class T1, class T2> using type = ops::add<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::min> {
    template <class T1, class T2> using type = ops::min<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::max> {
    template <class T1, class T2> using type = ops::max<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::sum> {
    template <class T1, class T2> using type = ops::add<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::prod> {
    template <class T1, class T2> using type = ops::mul<T1, T2>;
};

template <reduce_op Op, bool Accumulate, IsTensor TIn, IsTensor TOut,
          IsFixedDims Axes, IsFixedDims PackedAxes, class PadedNums>
class reduce_impl {
    using TInElem = typename TIn::element_type;
    using TOutElem = typename TOut::element_type;
    using TOutScalar = element_or_scalar_t<TOutElem>;

    static constexpr bool use_vector_reduce =
        PackedAxes::rank() == 1 && PackedAxes::at(0) >= Axes::at(0);

    static constexpr TOutElem initial_value() noexcept {
        if constexpr (Op == reduce_op::mean || Op == reduce_op::sum) {
            return (TOutElem)0;
        } else if constexpr (Op == reduce_op::min) {
            return (TOutElem)std::numeric_limits<TOutScalar>::max();
        } else if constexpr (Op == reduce_op::max) {
            return (TOutElem)std::numeric_limits<TOutScalar>::lowest();
        } else if constexpr (Op == reduce_op::prod) {
            return (TOutElem)1;
        }
    }

  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        auto in_p = input.elements().data();
        auto out_p = output.elements().data();
        // 1. Initialize
        if constexpr (!Accumulate) {
            ntt::apply(output.shape(),
                  [&](auto index) { output(index) = initial_value(); });
        }

        // 2. Reduce
        apply<0>(input, output, in_p, out_p);

        // 3. Mean
        if constexpr (Op == reduce_op::mean) {
            size_t inner_size =
                slice_fixed_dims<Axes::rank(), Axes::at(0)>(input.shape())
                    .length();
            if constexpr (use_vector_reduce) {
                inner_size *= TInElem::shape_type::length();
            }

            auto denom = (TOutScalar)inner_size;
            ntt::apply(output.shape(), [&](auto index) { output(index) /= denom; });
        }
    }

  private:
    template <size_t Axis, class TInP, class TOutP>
    constexpr void apply(const TIn &input, TOut &output, TInP in_p,
                         TOutP out_p) {
        for (size_t i = 0; i < input.shape()[Axis]; i++) {
            if constexpr (Axis == TIn::rank() - 1) {
                reduce(*out_p, *in_p);
            } else {
                apply<Axis + 1>(input, output, in_p, out_p);
            }

            in_p += input.strides()[Axis];
            out_p +=
                utility_detail::get_safe_stride(output, Axis, TOut::shape());
        }
    }

    template <class TOutElem, class TInElem>
    void reduce(TOutElem &output, const TInElem input) {
        if constexpr (IsScalar<TOutElem>) {
            output = ntt::reduce<reduce_to_binary_type<Op>::template type>(
                input, output);
        } else {
            output =
                reduce_to_binary_type<Op>::template type<TOutElem, TInElem>()(
                    output, input);
        }
    }
};
} // namespace detail

template <reduce_op Op, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums, class TIn, class TOut>
void reduce(const TIn &input, TOut &&output) noexcept {
    static_assert(PackedAxes::rank() < 2, "currently not support 2d packing.");

    static_assert(PadedNums::rank() == 0 ||
                      (PadedNums::rank() == 1 && PadedNums::at(0) == 0),
                  "not support padding");
    AUTO_NTT_PROFILER
    detail::reduce_impl<Op, false, std::decay_t<TIn>, std::decay_t<TOut>, Axes,
                        PackedAxes, PadedNums>
        impl;
    impl(input, output);
}

template <IsFixedDims Axes, IsFixedDims PackedAxes = fixed_shape<>,
          IsFixedDims PadedNums = fixed_shape<>, class TIn, class TOut>
void reduce_sum(const TIn &input, TOut &&output) noexcept {
    return reduce<reduce_op::sum, Axes, PackedAxes, PadedNums>(
        input, std::forward<TOut>(output));
}

template <IsFixedDims Axes, IsFixedDims PackedAxes = fixed_shape<>,
          IsFixedDims PadedNums = fixed_shape<>, class TIn, class TOut>
void reduce_min(const TIn &input, TOut &&output) noexcept {
    return reduce<reduce_op::min, Axes, PackedAxes, PadedNums>(
        input, std::forward<TOut>(output));
}

template <IsFixedDims Axes, IsFixedDims PackedAxes = fixed_shape<>,
          IsFixedDims PadedNums = fixed_shape<>, class TIn, class TOut>
void reduce_max(const TIn &input, TOut &&output) noexcept {
    return reduce<reduce_op::max, Axes, PackedAxes, PadedNums>(
        input, std::forward<TOut>(output));
}

template <IsFixedDims Axes, IsFixedDims PackedAxes = fixed_shape<>,
          IsFixedDims PadedNums = fixed_shape<>, class TIn, class TOut>
void reduce_mean(const TIn &input, TOut &&output) noexcept {
    return reduce<reduce_op::mean, Axes, PackedAxes, PadedNums>(
        input, std::forward<TOut>(output));
}
} // namespace nncase::ntt
