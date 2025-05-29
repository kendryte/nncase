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
#include "elementwise_impl.h"

namespace nncase::ntt::detail {
template <class TDerived, Tensor TLhs, Tensor TRhs, Tensor TOut>
class binary_like_impl
    : public elementwise_impl<true, TDerived, TOut, TLhs, TRhs> {
  public:
    using elementwise_impl<true, TDerived, TOut, TLhs, TRhs>::derived;

    template <size_t Axis, class TLhsP, class TRhsP, Shape TLhsRestShape,
              Shape TRhsRestShape, class TOutP, Shape TOutRestShape,
              class... TArgs>
    constexpr void apply_contiguous(TLhsP &lhs_p, TRhsP &rhs_p,
                                    const TLhsRestShape &lhs_rest_shape,
                                    const TRhsRestShape &rhs_rest_shape,
                                    TOutP &out_p, const TOutRestShape &,
                                    TArgs &&...args) {
        if (lhs_rest_shape.length() == 1) {
            return binary_left_broadcast(lhs_p, rhs_p, out_p,
                                         rhs_rest_shape.length(),
                                         std::forward<TArgs>(args)...);
        } else if (rhs_rest_shape.length() == 1) {
            return binary_right_broadcast(lhs_p, rhs_p, out_p,
                                          lhs_rest_shape.length(),
                                          std::forward<TArgs>(args)...);
        } else {
            // Non broadcast
            return binary_non_broadcast(lhs_p, rhs_p, out_p,
                                        lhs_rest_shape.length(),
                                        std::forward<TArgs>(args)...);
        }
    }

  private:
    template <class TLhsElem, class TRhsElem, class TOutElem, class... TArgs>
    constexpr void binary_non_broadcast(const TLhsElem *lhs,
                                        const TRhsElem *rhs, TOutElem *output,
                                        size_t extent, TArgs &&...args) {
        derived().invoke_ukernel(lhs, dim_one, rhs, dim_one, output, dim_one,
                                 extent, std::forward<TArgs>(args)...);
    }

    template <class TLhsElem, class TRhsElem, class TOutElem, class... TArgs>
    constexpr void binary_left_broadcast(const TLhsElem *lhs,
                                         const TRhsElem *rhs, TOutElem *output,
                                         size_t extent, TArgs &&...args) {
        derived().invoke_ukernel(lhs, dim_zero, rhs, dim_one, output, dim_one,
                                 extent, std::forward<TArgs>(args)...);
    }

    template <class TLhsElem, class TRhsElem, class TOutElem, class... TArgs>
    constexpr void binary_right_broadcast(const TLhsElem *lhs,
                                          const TRhsElem *rhs, TOutElem *output,
                                          size_t extent, TArgs &&...args) {
        derived().invoke_ukernel(lhs, dim_one, rhs, dim_zero, output, dim_one,
                                 extent, std::forward<TArgs>(args)...);
    }
};
} // namespace nncase::ntt::detail
