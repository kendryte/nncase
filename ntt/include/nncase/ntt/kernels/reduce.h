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
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"
#include <limits>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
namespace detail {
template <reduce_op Op, bool Accumulate, IsVector TIn, IsVector TOut,
          size_t Axis>
class inner_reduce_impl;

template <reduce_op Op, bool Accumulate, IsVector TIn, IsVector TOut>
class inner_reduce_impl<Op, Accumulate, TIn, TOut, 0> {
  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        auto count = input.shape()[0] - 1;
        auto begin = input.buffer().data();
        output = u_reduce<Op, TOut>(begin + 1, 1, count, *begin);
    }
};

template <reduce_op Op, bool Accumulate, IsVector TIn, IsVector TOut>
class inner_reduce_impl<Op, Accumulate, TIn, TOut, 1> {
    using TElem = typename TOut::element_type;

  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        for (size_t i = 0; i < output.shape()[0]; i++) {
            output(i) =
                ntt::reduce<ukernels::reduce_to_binary_type<Op>::template type,
                            TElem>(input(i));
        }
    }
};

template <reduce_op Op, bool Accumulate, IsTensor TIn, IsTensor TOut,
          IsFixedDims Axes, IsFixedDims PackedAxes, IsFixedDims PadedNums>
class reduce_impl {
    using TInElem = typename TIn::element_type;
    using TOutElem = typename TOut::element_type;
    using TOutScalar = element_or_scalar_t<TOutElem>;

    static constexpr TInElem initial_value() noexcept {
        if constexpr (Op == reduce_op::mean || Op == reduce_op::sum) {
            return (TInElem)0;
        } else if constexpr (Op == reduce_op::min) {
            return (TInElem)std::numeric_limits<TOutScalar>::max();
        } else if constexpr (Op == reduce_op::max) {
            return (TInElem)std::numeric_limits<TOutScalar>::lowest();
        } else if constexpr (Op == reduce_op::prod) {
            return (TInElem)1;
        }
    }

  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        ntt::apply(output.shape(), [&](auto index) {
            auto reduced_in = (TInElem)initial_value();
            apply_reduce(input, reduce_source_offset<TIn::rank(), Axes>(index),
                         reduced_in);
            if constexpr (IsScalar<TOutElem>) {
                output(index) = ntt::reduce<
                    ukernels::reduce_to_binary_type<Op>::template type,
                    TOutElem>(reduced_in);
            } else if constexpr (IsTensor<TOutElem> && TInElem::rank() == 2 &&
                                 TOutElem::rank() == 1) {
                constexpr auto inner_axis =
                    PackedAxes::at(0) == Axes::at(0) ? 0 : 1;
                inner_reduce_impl<Op, false, TInElem, TOutElem, inner_axis>
                    inner_impl;
                inner_impl(reduced_in, output(index));
            } else {
                output(index) = reduced_in;
            }

            // Mean
            if constexpr (Op == reduce_op::mean) {
                size_t inner_size =
                    slice_fixed_dims<Axes::rank(), Axes::at(0)>(input.shape())
                        .length();
                if constexpr (IsVector<TOutElem>) {
                    inner_size *= TInElem::shape_type::length();
                }

                if constexpr (IsVector<TOutElem>) {
                    inner_size /= TOutElem::shape_type::length();
                }

                auto denom = (TOutScalar)inner_size;
                output(index) /= denom;
            }
        });
    }

  private:
    constexpr void apply_reduce(const TIn &input,
                                ranked_shape<TIn::rank()> index,
                                TInElem &reduced_in) {
        auto src_tensor =
            input.view(index, fixed_reduce_source_shape_type<Axes, TIn>());
        auto conti_dims =
            contiguous_dims(src_tensor.shape(), src_tensor.strides());
        if (conti_dims > 1) {
            ranked_shape<TIn::rank()> src_index{};
            apply_contiguous_reduce<0>(src_index, conti_dims, src_tensor,
                                       reduced_in);
        } else {
            apply_non_contiguous_reduce<0>(input, index, reduced_in);
        }
    }

    template <size_t Axis, class TSubIn>
    constexpr void apply_contiguous_reduce(ranked_shape<TSubIn::rank()> &index,
                                           size_t conti_dims,
                                           const TSubIn &input,
                                           TInElem &reduced_in) {
        const auto outer_dims = TSubIn::rank() - conti_dims;
        if (Axis >= outer_dims) {
            size_t inner_size = 1;
            for (size_t i = outer_dims; i < input.shape().rank(); i++)
                inner_size *= input.shape()[i];
            auto input_p =
                input.buffer().data() + linear_offset(index, input.strides());
            reduced_in = ntt::u_reduce<Op>(input_p, 1, inner_size, reduced_in);
        } else if constexpr (Axis < TSubIn::rank() - 1) {
            const auto dim = input.shape()[Axis];
            for (index[Axis] = 0; index[Axis] < dim; index[Axis]++) {
                apply_contiguous_reduce<Axis + 1>(index, conti_dims, input,
                                                  reduced_in);
            }
        }
    }

    template <size_t ReduceIndex>
    constexpr void apply_non_contiguous_reduce(const TIn &input,
                                               ranked_shape<TIn::rank()> index,
                                               TInElem &reduced_in) {
        constexpr size_t Axis = Axes::at(ReduceIndex);
        if constexpr (ReduceIndex < Axes::rank() - 1) {
            for (size_t i = 0; i < input.shape()[Axis]; i++) {
                index[Axis] = i;
                apply_non_contiguous_reduce<ReduceIndex + 1>(input, index,
                                                             reduced_in);
            }
        } else {
            const TInElem *in_p = &input(index);
            reduced_in = ntt::u_reduce<Op>(in_p, input.strides()[Axis],
                                           input.shape()[Axis], reduced_in);
        }
    }
};
} // namespace detail

template <reduce_op Op, IsFixedDims Axes, IsFixedDims PackedAxes,
          IsFixedDims PadedNums, class TIn, class TOut>
void reduce(const TIn &input, TOut &&output) noexcept {
    static_assert(PadedNums::rank() == 0 ||
                      (PadedNums::rank() == 1 && PadedNums::at(0) == 0) ||
                      (PadedNums::rank() == 2 && PadedNums::at(0) == 0 &&
                       PadedNums::at(1) == 0),
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
