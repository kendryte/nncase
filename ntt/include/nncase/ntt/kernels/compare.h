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
#include "../ukernels.h"
#include "../utility.h"
#include "nncase/ntt/shape.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TLhs, Tensor TRhs, Tensor TOut> class compare_impl {
  public:
    template <class Op>
    constexpr void operator()(Op &op, const TLhs &lhs, const TRhs &rhs,
                              TOut &output) {
        const auto conti_dims =
            ntt::min(contiguous_dims(TLhs::shape(), TLhs::strides()),
                     contiguous_dims(TRhs::shape(), TRhs::strides()),
                     contiguous_dims(TOut::shape(), TOut::strides()));
        auto lhs_p = lhs.elements().data();
        auto rhs_p = rhs.elements().data();
        auto out_p = output.elements().data();
        apply<Op, 0>(op, conti_dims, lhs, rhs, output, lhs_p, rhs_p, out_p);
    }

  private:
    template <class Op, size_t Axis, Dimension TContiguousDims, class TLhsP,
              class TRhsP, class TOutP>
    constexpr void apply(Op &op, const TContiguousDims &conti_dims,
                         const TLhs &lhs, const TRhs &rhs, TOut &output,
                         TLhsP lhs_p, TRhsP rhs_p, TOutP out_p) {
        // 1. In contiguous axes
        if (Axis + conti_dims >= TOut::rank()) {
            const auto rest_rank = TOut::rank() - Axis;
            const auto lhs_rest_dims =
                lhs.shape().slice(rest_rank, TLhs::rank() - rest_rank);
            const auto rhs_rest_dims =
                rhs.shape().slice(rest_rank, TRhs::rank() - rest_rank);

            // 1.1 Non broadcast
            if (lhs_rest_dims == rhs_rest_dims) {
                return compare_non_broadcast<Op>(lhs_p, rhs_p, out_p,
                                                 lhs_rest_dims.length());
            } else if (lhs_rest_dims.length() == 1) {
                return compare_left_broadcast<Op>(lhs_p, rhs_p, out_p,
                                                  rhs_rest_dims.length());
            } else if (rhs_rest_dims.length() == 1) {
                return compare_right_broadcast<Op>(lhs_p, rhs_p, out_p,
                                                   lhs_rest_dims.length());
            }
        }

        // 2. Out of contiguous axes
        if constexpr (Axis < TOut::shape().rank()) {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply<Op, Axis + 1>(op, conti_dims, lhs, rhs, output, lhs_p,
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
    void compare_non_broadcast(const TLhsElem *lhs, const TRhsElem *rhs,
                               TOutElem *output, size_t extent) {
        ntt::u_compare<Op, TLhsElem, TRhsElem, TOutElem>(lhs, 1, rhs, 1, output,
                                                         1, extent);
    }

    template <class Op, class TLhsElem, class TRhsElem, class TOutElem>
    void compare_left_broadcast(const TLhsElem *lhs, const TRhsElem *rhs,
                                TOutElem *output, size_t extent) {
        ntt::u_compare<Op, TLhsElem, TRhsElem, TOutElem>(lhs, 0, rhs, 1, output,
                                                         1, extent);
    }

    template <class Op, class TLhsElem, class TRhsElem, class TOutElem>
    void compare_right_broadcast(const TLhsElem *lhs, const TRhsElem *rhs,
                                 TOutElem *output, size_t extent) {
        ntt::u_compare<Op, TLhsElem, TRhsElem, TOutElem>(lhs, 1, rhs, 0, output,
                                                         1, extent);
    }
};
} // namespace detail

template <template <class T1, class T2> class Op, class TLhs, class TRhs,
          class TOut>
void compare(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    static_assert(std::is_same_v<typename TLhs::element_type,
                                 typename TRhs::element_type>,
                  "compare only support same element type now");
    Op<typename TLhs::element_type, typename TRhs::element_type> op;
    detail::compare_impl<std::decay_t<TLhs>, std::decay_t<TRhs>,
                         std::decay_t<TOut>>()(op, lhs, rhs, output);
}
} // namespace nncase::ntt
