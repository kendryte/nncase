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
#include "detail/elementwise_impl.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TCond, Tensor TX, Tensor TY, Tensor TOut>
class where_impl : public elementwise_impl<where_impl<TCond, TX, TY, TOut>,
                                           TOut, TCond, TX, TY> {
  public:
    template <Tensor TBroadcastedCond, Tensor TBroadcastedX,
              Tensor TBroadcastedY>
    constexpr void apply(const TBroadcastedCond &cond, const TBroadcastedX &x,
                         const TBroadcastedY &y, TOut &output) {
        ntt::apply(output.shape(), [&](auto index) {
            output(index) = ntt::where(cond(index), x(index), y(index));
        });
    }
};
} // namespace detail

template <Tensor TCond, Tensor TX, Tensor TY, class TOut>
void where(const TCond &cond, const TX &x, const TY &y, TOut &&output) {
    detail::where_impl<TCond, TX, TY, std::decay_t<TOut>>()(cond, x, y, output);
}
} // namespace nncase::ntt
