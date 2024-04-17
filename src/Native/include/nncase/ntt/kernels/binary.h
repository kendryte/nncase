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
#include "../shape_infer/binary.h"
#include "../shape_infer/reduce.h"
#include "../tensor_traits.h"
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
        constexpr auto out_shape =
            shape_infer::binary_output_shape(TLhs::shape(), TRhs::shape());
        auto lhs_p = lhs.buffer().data();
        auto rhs_p = rhs.buffer().data();
        auto out_p = output.buffer().data();
        apply<Op, 0, conti_dims>(op, out_shape, lhs, rhs, output, lhs_p, rhs_p,
                                 out_p);
    }

  private:
    template <class Op, size_t Axis, size_t ContiguousDims, class TOutShape,
              class TLhsP, class TRhsP, class TOutP>
    constexpr void apply(Op &op, const TOutShape &out_shape, const TLhs &lhs,
                         const TRhs &rhs, TOut &output, TLhsP lhs_p,
                         TRhsP rhs_p, TOutP out_p) {
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
        if constexpr (Axis < out_shape.rank()) {
            for (size_t i = 0; i < out_shape[Axis]; i++) {
                apply<Op, Axis + 1, ContiguousDims>(
                    op, out_shape, lhs, rhs, output, lhs_p, rhs_p, out_p);
                lhs_p += get_safe_stride(lhs, Axis, out_shape);
                rhs_p += get_safe_stride(rhs, Axis, out_shape);
                out_p += output.strides()[Axis];
            }
        }
    }

    template <class TTensor, class TOutShape>
    static constexpr size_t
    get_safe_stride(const TTensor &tensor, size_t axis,
                    const TOutShape &out_shape) noexcept {
        auto dim_ext = out_shape.rank() - tensor.rank();
        if (axis < dim_ext) {
            return 0;
        }

        auto actual_axis = axis - dim_ext;
        return tensor.shape()[actual_axis] == 1 ? 0 // broadcast
                                                : tensor.strides()[actual_axis];
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
} // namespace detail

template <template <class T1, class T2> class Op, class TLhs, class TRhs,
          class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &output) {
    Op<typename TLhs::element_type, typename TRhs::element_type> op;
    detail::binary_impl<std::decay_t<TLhs>, std::decay_t<TRhs>,
                        std::decay_t<TOut>>()(op, lhs, rhs, output);
}
} // namespace nncase::ntt
