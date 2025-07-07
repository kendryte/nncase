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
#include "detail/unary_like_impl.h"

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut>
class unary_impl : public unary_like_impl<unary_impl<TIn, TOut>, TIn, TOut> {
  public:
    template <Tensor TBroadcastedIn, class Op>
    void invoke_ukernel(const TBroadcastedIn &input, TOut &output,
                        const Op &op) {
        ntt::apply(output.shape(),
                   [&](auto index) { output(index) = op(input(index)); });
    }
};
} // namespace detail

template <template <class T> class Op, Tensor TIn, class TOut>
void unary(const TIn &input, TOut &&output,
           const Op<std::remove_cv_t<typename TIn::element_type>> &op = {}) {
    detail::unary_impl<TIn, std::decay_t<TOut>> impl;
    impl(input, output, op);
}
} // namespace nncase::ntt
