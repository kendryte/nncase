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
#include "detail/unary_like_impl.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut>
class expand_impl : public unary_like_impl<expand_impl<TIn, TOut>, TIn, TOut> {
    using TElem = typename TIn::element_type;

  public:
    template <Tensor TBroadcastedIn>
    void invoke_ukernel(const TBroadcastedIn &input, TOut &output) {
        ntt::apply(output.shape(),
                   [&](auto index) { output(index) = input(index); });
    }
};

} // namespace detail

template <Tensor TIn, typename TOut>
void expand(const TIn &input, TOut &&output) noexcept {
    detail::expand_impl<TIn, std::decay_t<TOut>> impl;
    impl(input, output);
}
} // namespace nncase::ntt
