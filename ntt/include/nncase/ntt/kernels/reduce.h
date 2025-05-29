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
#include "../primitive_ops.h"
#include "../shape_infer/reduce.h"
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/tensor_traits.h"
#include <limits>

namespace nncase::ntt {
namespace detail {
template <reduce_op Op, bool Accumulate, Vector TIn, Vector TOut, size_t Axis>
class inner_reduce_impl;

template <reduce_op Op, bool Accumulate, Vector TIn, Vector TOut>
class inner_reduce_impl<Op, Accumulate, TIn, TOut, 0> {
  public:
    constexpr void operator()(const TIn &input, TOut &output) {
        auto count = input.shape()[0] - 1;
        auto begin = input.buffer().data();
        output = u_reduce<Op, TOut>(begin + 1, 1, count, *begin);
    }
};

template <reduce_op Op, bool Accumulate, Vector TIn, Vector TOut>
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

template <reduce_op Op, bool LoadPrevious, Tensor TIn, Tensor TOut,
          FixedDimensions PadedNums>
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
    template <FixedDimensions TReduceAxes, FixedDimensions PackedAxes>
    constexpr void operator()(const TIn &input, TOut &output,
                              const TReduceAxes &reduce_axes,
                              const PackedAxes &packed_axes) {
        ntt::apply(output.shape(), [&](auto index) {
            auto reduced_in = (TInElem)initial_value();
            auto source_index =
                shape_infer::reduce_source_index_template<TIn::rank()>(
                    index, reduce_axes);
            apply_reduce(input, source_index, reduced_in, reduce_axes);
            TOutElem reduced_out;
            if constexpr (Scalar<TOutElem>) {
                reduced_out = ntt::reduce<
                    ukernels::reduce_to_binary_type<Op>::template type,
                    TOutElem>(reduced_in);
            } else if constexpr (Vector<TOutElem> && TInElem::rank() == 2 &&
                                 TOutElem::rank() == 1) {
                constexpr auto inner_axis =
                    ntt::select<(packed_axes.at(0) == reduce_axes.at(0))>(
                        0_dim, 1_dim);
                inner_reduce_impl<Op, false, TInElem, TOutElem, inner_axis>
                    inner_impl;
                inner_impl(reduced_in, reduced_out);
            } else {
                reduced_out = reduced_in;
            }

            if constexpr (LoadPrevious) {
                output(index) = ntt::reduce<
                    ukernels::reduce_to_binary_type<Op>::template type>(
                    reduced_out, output(index));
            } else {
                output(index) = reduced_out;
            }

            // Mean
            if constexpr (Op == reduce_op::mean) {
                constexpr auto reduce_axis = reduce_axes[0_dim];
                const auto inner_size =
                    input.shape()
                        .template slice<reduce_axis, TReduceAxes::rank()>()
                        .length();
                auto denom = (TOutScalar)inner_size;
                if constexpr (Vector<TOutElem>) {
                    const auto inner_size_unpacked = inner_size *
                                                     TInElem::shape().length() /
                                                     TOutElem::shape().length();
                    denom = (TOutScalar)inner_size_unpacked;
                }
                output(index) /= denom;
            }
        });
    }

  private:
    template <Shape TInIndex, FixedDimensions TReduceAxes>
    constexpr void apply_reduce(const TIn &input, TInIndex &index,
                                TInElem &reduced_in,
                                const TReduceAxes &reduce_axes) {
        auto src_tensor = input.view(
            index,
            shape_infer::sub_reduce_source_shape(input.shape(), reduce_axes));
        const auto conti_dims =
            contiguous_dims(src_tensor.shape(), src_tensor.strides());
        if (conti_dims > 1) {
            dynamic_shape_t<TIn::rank()> src_index{};
            apply_contiguous_reduce<0>(src_index, conti_dims, src_tensor,
                                       reduced_in);
        } else {
            apply_non_contiguous_reduce<0>(input, index, reduced_in,
                                           reduce_axes);
        }
    }

    template <size_t Axis, class TSubIn>
    constexpr void
    apply_contiguous_reduce(dynamic_shape_t<TSubIn::rank()> &index,
                            size_t conti_dims, const TSubIn &input,
                            TInElem &reduced_in) {
        const auto outer_dims = TSubIn::rank() - conti_dims;
        const auto axis_v = fixed_dim_v<Axis>;
        if (Axis >= outer_dims) {
            size_t inner_size = 1;
            for (size_t i = outer_dims; i < input.shape().rank(); i++)
                inner_size *= input.shape()[i];
            auto input_p =
                input.buffer().data() + linear_offset(index, input.strides());
            reduced_in = ntt::u_reduce<Op>(input_p, 1, inner_size, reduced_in);
        } else if constexpr (Axis < TSubIn::rank() - 1) {
            const auto dim = input.shape()[Axis];
            for (index[axis_v] = 0; index[axis_v] < dim; index[axis_v]++) {
                apply_contiguous_reduce<Axis + 1>(index, conti_dims, input,
                                                  reduced_in);
            }
        }
    }

    template <size_t ReduceIndex, Shape TInIndex, FixedDimensions TReduceAxes>
    constexpr void apply_non_contiguous_reduce(const TIn &input,
                                               TInIndex &index,
                                               TInElem &reduced_in,
                                               const TReduceAxes &reduce_axes) {
        const auto axis = reduce_axes[fixed_dim_v<ReduceIndex>];
        if constexpr (ReduceIndex < TReduceAxes::rank() - 1) {
            for (size_t i = 0; i < input.shape()[axis]; i++) {
                index[axis] = i;
                apply_non_contiguous_reduce<ReduceIndex + 1>(input, index,
                                                             reduced_in);
            }
        } else {
            const TInElem *in_p = &input(index);
            reduced_in = ntt::u_reduce<Op>(in_p, input.strides()[axis],
                                           input.shape()[axis], reduced_in);
        }
    }
};
} // namespace detail

template <reduce_op Op, bool LoadPrevious = false, Tensor TIn, class TOut,
          FixedDimensions TReduceAxes, FixedDimensions PackedAxes = shape_t<>,
          FixedDimensions PadedNums =
              decltype(make_zeros_shape<PackedAxes::rank()>())>
void reduce(const TIn &input, TOut &&output,
            [[maybe_unused]] const TReduceAxes &reduce_axes,
            [[maybe_unused]] const PackedAxes &packed_axes = {},
            [[maybe_unused]] const PadedNums &paded_nums = {}) noexcept {
    static_assert(paded_nums == make_zeros_shape<PackedAxes::rank()>(),
                  "not support padding");
    static_assert(!(LoadPrevious && Op == reduce_op::mean),
                  "not support reduce mean splited on reduce axis");
    detail::reduce_impl<Op, LoadPrevious, TIn, std::decay_t<TOut>, PadedNums>
        impl;
    impl(input, output, reduce_axes, packed_axes);
}

#define DEFINE_NTT_REDUCE(op)                                                  \
    template <bool LoadPrevious = false, Tensor TIn, class TOut,               \
              FixedDimensions TReduceAxes,                                     \
              FixedDimensions PackedAxes = shape_t<>,                          \
              FixedDimensions PadedNums =                                      \
                  decltype(make_zeros_shape<PackedAxes::rank()>())>            \
    void reduce_##op(const TIn &input, TOut &&output,                          \
                     const TReduceAxes &reduce_axes,                           \
                     const PackedAxes &packed_axes = {},                       \
                     const PadedNums &paded_nums = {}) noexcept {              \
        return reduce<reduce_op::op, LoadPrevious>(                            \
            input, std::forward<TOut>(output), reduce_axes, packed_axes,       \
            paded_nums);                                                       \
    }

DEFINE_NTT_REDUCE(mean)
DEFINE_NTT_REDUCE(min)
DEFINE_NTT_REDUCE(max)
DEFINE_NTT_REDUCE(sum)

#undef DEFINE_NTT_REDUCE
} // namespace nncase::ntt
