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
#include "../shape_infer/binary.h"
#include "../shape_infer/reduce.h"
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../ukernels.h"
#include "../utility.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TLhs, class TRhs, class TOut> class binary_impl {
  public:
    template <class Op>
    constexpr void operator()(Op &op, const TLhs &lhs, const TRhs &rhs,
                              TOut &output) {
        auto out_shape =
            shape_infer::binary_output_shape(lhs.shape(), rhs.shape());

        apply(out_shape, [&](auto index) {
            const auto lhs_index =
                shape_infer::reduced_index_by_shape(index, lhs.shape());
            const auto rhs_index =
                shape_infer::reduced_index_by_shape(index, rhs.shape());
            output(index) = op(lhs(lhs_index), rhs(rhs_index));
        });
    }
};

template <IsFixedTensor TLhs, IsFixedTensor TRhs, IsFixedTensor TOut>
class binary_impl<TLhs, TRhs, TOut> {
  public:
    template <class Op>
    constexpr void operator()(Op &op, const TLhs &lhs, const TRhs &rhs,
                              TOut &output) {
        constexpr auto conti_dims =
            std::min({contiguous_dims(TLhs::shape(), TLhs::strides()),
                      contiguous_dims(TRhs::shape(), TRhs::strides()),
                      contiguous_dims(TOut::shape(), TOut::strides())});
        auto lhs_p = lhs.elements().data();
        auto rhs_p = rhs.elements().data();
        auto out_p = output.elements().data();
        apply<Op, 0, conti_dims>(op, lhs, rhs, output, lhs_p, rhs_p, out_p);
    }

  private:
    template <class Op, size_t Axis, size_t ContiguousDims, class TLhsP,
              class TRhsP, class TOutP>
    constexpr void apply(Op &op, const TLhs &lhs, const TRhs &rhs, TOut &output,
                         TLhsP lhs_p, TRhsP rhs_p, TOutP out_p) {
        // 1. In contiguous axes
        if constexpr (Axis + ContiguousDims >= TOut::rank()) {
            constexpr auto rest_rank = TOut::rank() - Axis;
            constexpr auto lhs_rest_dims =
                slice_fixed_dims<rest_rank, TLhs::rank() - rest_rank>(
                    TLhs::shape());
            constexpr auto rhs_rest_dims =
                slice_fixed_dims<rest_rank, TRhs::rank() - rest_rank>(
                    TRhs::shape());

            // 1.1 Non broadcast
            if constexpr (is_same_seq(lhs_rest_dims, rhs_rest_dims)) {
                return binary_non_broadcast(op, lhs_p, rhs_p, out_p,
                                            lhs_rest_dims.length());
            } else if constexpr (lhs_rest_dims.length() == 1) {
                return binary_left_broadcast(op, *lhs_p, rhs_p, out_p,
                                             rhs_rest_dims.length());
            } else if constexpr (rhs_rest_dims.length() == 1) {
                return binary_right_broadcast(op, lhs_p, *rhs_p, out_p,
                                              lhs_rest_dims.length());
            }
        }

        // 2. Out of contiguous axes
        if constexpr (Axis < TOut::shape().rank()) {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply<Op, Axis + 1, ContiguousDims>(op, lhs, rhs, output, lhs_p,
                                                    rhs_p, out_p);
                lhs_p +=
                    utility_detail::get_safe_stride(lhs, Axis, TOut::shape());
                rhs_p +=
                    utility_detail::get_safe_stride(rhs, Axis, TOut::shape());
                out_p += output.strides()[Axis];
            }
        }
    }

    template <class Op, class TLhsElem, class TRhsElem, class TOutElem>
    void binary_non_broadcast(Op &op, const TLhsElem *lhs, const TRhsElem *rhs,
                              TOutElem *output, size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            *output++ = op(*lhs++, *rhs++);
        }
    }

    template <class Op, class TLhsElem, class TRhsElem, class TOutElem>
    void binary_left_broadcast(Op &op, const TLhsElem &lhs, const TRhsElem *rhs,
                               TOutElem *output, size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            *output++ = op(lhs, *rhs++);
        }
    }

    template <class Op, class TLhsElem, class TRhsElem, class TOutElem>
    void binary_right_broadcast(Op &op, const TLhsElem *lhs,
                                const TRhsElem &rhs, TOutElem *output,
                                size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            *output++ = op(*lhs++, rhs);
        }
    }
};

#define BINARY_IMPL(OP)                                                        \
    template <class Shape, class In1Strides, class In2Strides,                 \
              class OutStrides>                                                \
    class OP##_impl;                                                           \
    template <size_t... Dims, size_t... In1Strides, size_t... In2Strides,      \
              size_t... OutStrides>                                            \
    class OP##_impl<fixed_shape<Dims...>, fixed_strides<In1Strides...>,        \
                    fixed_strides<In2Strides...>,                              \
                    fixed_strides<OutStrides...>> {                            \
      public:                                                                  \
        template <class TIn1, class TIn2, class TOut>                          \
        constexpr void operator()(const TIn1 &input1, const TIn2 &input2,      \
                                  TOut &output) {                              \
            constexpr size_t rank = sizeof...(Dims);                           \
            ranked_shape<rank> index{};                                        \
            constexpr auto conti_dims =                                        \
                std::min(contiguous_dims(fixed_shape<Dims...>{},               \
                                         fixed_strides<In1Strides...>{}),      \
                         contiguous_dims(fixed_shape<Dims...>{},               \
                                         fixed_strides<OutStrides...>{}));     \
            apply<TIn1, TIn2, TOut, 0, rank, conti_dims, Dims...>(             \
                index, input1, input2, output);                                \
        }                                                                      \
                                                                               \
      private:                                                                 \
        template <class TIn1, class TIn2, class TOut, size_t Axis,             \
                  size_t Rank, size_t ContiguousDims, size_t... RestDims>      \
        constexpr void apply(ranked_shape<Rank> &index, const TIn1 &input1,    \
                             const TIn2 &input2, TOut &output) {               \
            if constexpr (ContiguousDims == sizeof...(RestDims)) {             \
                constexpr auto inner_size =                                    \
                    fixed_shape<RestDims...>::length();                        \
                auto input1_p = input1.elements().data() +                     \
                                linear_offset(index, input1.strides());        \
                auto input2_p = input2.elements().data() +                     \
                                linear_offset(index, input2.strides());        \
                auto output_p = output.elements().data() +                     \
                                linear_offset(index, output.strides());        \
                OP##_contiguous<inner_size>(input1_p, input2_p, output_p);     \
            } else {                                                           \
                apply_next<TIn1, TIn2, TOut, Axis, Rank, ContiguousDims,       \
                           RestDims...>(index, input1, input2, output);        \
            }                                                                  \
        }                                                                      \
                                                                               \
        template <class TIn1, class TIn2, class TOut, size_t Axis,             \
                  size_t Rank, size_t ContiguousDims, size_t Dim,              \
                  size_t... RestDims>                                          \
        constexpr void apply_next(ranked_shape<Rank> &index,                   \
                                  const TIn1 &input1, const TIn2 &input2,      \
                                  TOut &output) {                              \
            for (index[Axis] = 0; index[Axis] < Dim; index[Axis]++) {          \
                apply<TIn1, TIn2, TOut, Axis + 1, Rank, ContiguousDims,        \
                      RestDims...>(index, input1, input2, output);             \
            }                                                                  \
        }                                                                      \
        template <size_t Extent, class T1, class T2, class TOut>               \
        constexpr void OP##_contiguous(const T1 *input1, const T2 *input2,     \
                                       TOut *output) {                         \
            ntt::u_##OP(input1, input2, 1, 1, output, 1, Extent);              \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <size_t Rank, class In1Strides, class In2Strides,                 \
              class OutStrides>                                                \
    class OP##_impl<ranked_shape<Rank>, In1Strides, In2Strides, OutStrides> {  \
      public:                                                                  \
        template <class TIn1, class TIn2, class TOut>                          \
        constexpr void operator()(const TIn1 &input1, const TIn2 &input2,      \
                                  TOut &output) {                              \
            ranked_shape<Rank> index{};                                        \
            auto conti_dims =                                                  \
                std::min(contiguous_dims(input1.shape(), input1.strides()),    \
                         contiguous_dims(input1.shape(), output.strides()));   \
            apply<TIn1, TIn2, TOut, 0>(index, conti_dims, input1, input2,      \
                                       output);                                \
        }                                                                      \
                                                                               \
      private:                                                                 \
        template <class TIn1, class TIn2, class TOut, size_t Axis>             \
        constexpr void apply(ranked_shape<Rank> &index, size_t conti_dims,     \
                             const TIn1 &input1, const TIn2 &input2,           \
                             TOut &output) {                                   \
            const auto outer_dims = Rank - conti_dims;                         \
            if (Axis >= outer_dims) {                                          \
                size_t inner_size = 1;                                         \
                for (size_t i = outer_dims; i < input1.shape().rank(); i++)    \
                    inner_size *= input1.shape()[i];                           \
                auto input1_p = input1.buffer().data() +                       \
                                linear_offset(index, input1.strides());        \
                auto input2_p = input2.buffer().data() +                       \
                                linear_offset(index, input2.strides());        \
                auto output_p = output.buffer().data() +                       \
                                linear_offset(index, output.strides());        \
                OP##_contiguous(input1_p, input2_p, output_p, inner_size);     \
            } else if constexpr (Axis < Rank - 1) {                            \
                const auto dim = input1.shape()[Axis];                         \
                for (index[Axis] = 0; index[Axis] < dim; index[Axis]++) {      \
                    apply<TIn1, TIn2, TOut, Axis + 1>(index, conti_dims,       \
                                                      input1, input2, output); \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        template <class T1, class T2, class TOut>                              \
        constexpr void OP##_contiguous(const T1 *input1_p, const T2 *input2_p, \
                                       TOut *output_p, size_t extent) {        \
            for (size_t i = 0; i < extent; i++) {                              \
                output_p[i] =                                                  \
                    ntt::ops::OP<T1, T2>()(input1_p[i], input2_p[i]);          \
            }                                                                  \
        }                                                                      \
    };

BINARY_IMPL(add)
BINARY_IMPL(div)
BINARY_IMPL(max)
BINARY_IMPL(min)
BINARY_IMPL(mod)
BINARY_IMPL(mul)
BINARY_IMPL(sub)

} // namespace detail

template <template <class T1, class T2> class Op, class TLhs, class TRhs,
          class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    Op<typename TLhs::element_type, typename TRhs::element_type> op;
    detail::binary_impl<std::decay_t<TLhs>, std::decay_t<TRhs>,
                        std::decay_t<TOut>>()(op, lhs, rhs, output);
}

#define BINARY(OP)                                                             \
    template <typename TIn1, typename TIn2, typename TOut>                     \
    void OP(const TIn1 &input1, const TIn2 &input2, TOut &&output) noexcept {  \
        detail::OP##_impl<                                                     \
            common_shape_t<typename TIn1::shape_type,                          \
                           typename std::decay_t<TOut>::shape_type>,           \
            typename TIn1::strides_type, typename TIn2::strides_type,          \
            typename std::decay_t<TOut>::strides_type>                         \
            impl;                                                              \
        impl(input1, input2, output);                                          \
    }

BINARY(add)
BINARY(div)
BINARY(max)
BINARY(min)
BINARY(mod)
BINARY(mul)
BINARY(sub)

} // namespace nncase::ntt
