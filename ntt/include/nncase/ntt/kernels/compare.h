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
class compare_impl : public binary_like_impl<compare_impl<TLhs, TRhs, TOut>,
                                             TLhs, TRhs, TOut> {
  public:
    template <Tensor TBroadcastedLhs, Tensor TBroadcastedRhs, class Op>
    void invoke_ukernel(const TBroadcastedLhs &lhs, const TBroadcastedRhs &rhs,
                        TOut &output, const Op &op) {
        ntt::apply(output.shape(), [&](auto index) {
            output(index) = op(lhs(index), rhs(index));
        });
    }
};
} // namespace detail

template <template <class T1, class T2> class Op, Tensor TLhs, Tensor TRhs,
          class TOut>
void compare(const TLhs &lhs, const TRhs &rhs, TOut &&output,
             const Op<std::remove_cv_t<typename TLhs::element_type>,
                      std::remove_cv_t<typename TRhs::element_type>> &op = {}) {
    // static_assert(std::is_same_v<typename TLhs::element_type,
    //                              typename TRhs::element_type>,
    //               "compare only support same element type now");
    detail::compare_impl<TLhs, TRhs, std::decay_t<TOut>>()(lhs, rhs, output,
                                                           op);
}
} // namespace nncase::ntt
