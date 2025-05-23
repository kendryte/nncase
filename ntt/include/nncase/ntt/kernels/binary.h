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
#include "../ukernels.h"
#include "detail/binary_like_impl.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TLhs, Tensor TRhs, Tensor TOut>
class binary_impl
    : public binary_like_impl<binary_impl<TLhs, TRhs, TOut>, TLhs, TRhs, TOut> {
  public:
    template <class TLhsElem, Dimension TLhsStride, class TRhsElem,
              Dimension TRhsStride, class TOutElem, Dimension TOutStride,
              class Op>
    void invoke_ukernel(const TLhsElem *lhs, const TLhsStride &lhs_stride,
                        const TRhsElem *rhs, const TRhsStride &rhs_stride,
                        TOutElem *output, const TOutStride &output_stride,
                        size_t extent, Op &op) {
        ntt::u_binary(op, lhs, lhs_stride, rhs, rhs_stride, output,
                      output_stride, extent);
    }
};
} // namespace detail

template <template <class T1, class T2> class Op, Tensor TLhs, Tensor TRhs,
          class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &&output,
            const Op<typename TLhs::element_type, typename TRhs::element_type>
                &op = {}) {
    detail::binary_impl<TLhs, TRhs, std::decay_t<TOut>>()(lhs, rhs, output, op);
}
} // namespace nncase::ntt
