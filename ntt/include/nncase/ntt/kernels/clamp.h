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
#include "../primitive_ops.h"
#include "../ukernels.h"
#include "detail/unary_like_impl.h"

namespace nncase::ntt {

namespace detail {
template <Tensor TIn, Tensor TOut, Scalar TElem>
class clamp_impl
    : public unary_like_impl<clamp_impl<TIn, TOut, TElem>, TIn, TOut> {

  public:
    template <Tensor TBroadcastedIn>
    void invoke_ukernel(const TBroadcastedIn &input, TOut &output,
                        const TElem &min, const TElem &max) {
        ntt::apply(output.shape(), [&](auto index) {
            output(index) = ntt::clamp(input(index), min, max);
        });
    }
};
} // namespace detail

template <Tensor TIn, class TOut, Scalar TElem>
void clamp(const TIn &input, TOut &&output, const TElem &min,
           const TElem &max) noexcept {
    detail::clamp_impl<TIn, std::decay_t<TOut>, TElem> impl;
    impl(input, output, min, max);
}
} // namespace nncase::ntt
