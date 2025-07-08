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
class binary_like_impl : public elementwise_impl<TDerived, TOut, TLhs, TRhs> {
  public:
    using elementwise_impl<TDerived, TOut, TLhs, TRhs>::derived;

    template <Tensor TBroadcastedLhs, Tensor TBroadcastedRhs, class... TArgs>
    constexpr void apply(const TBroadcastedLhs &lhs, const TBroadcastedRhs &rhs,
                         TOut &output, TArgs &&...args) {
        derived().invoke_ukernel(lhs, rhs, output,
                                 std::forward<TArgs>(args)...);
    }
};
} // namespace nncase::ntt::detail
