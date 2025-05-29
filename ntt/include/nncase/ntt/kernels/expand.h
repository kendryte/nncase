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
#include "../ukernels/u_unary.h"
#include "detail/elementwise_impl.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <Tensor TIn, Tensor TOut>
class expand_impl
    : public elementwise_impl<true, expand_impl<TIn, TOut>, TOut, TIn> {
    using TElem = typename TIn::element_type;

  public:
    template <size_t Axis, class TInP, Shape TInRestShape, class TOutP,
              Shape TOutRestShape>
    constexpr void
    apply_contiguous(TInP &in_p, const TInRestShape &in_rest_shape,
                     TOutP &out_p, const TOutRestShape &out_rest_shape) {
        if (in_rest_shape.length() == 1) {
            ntt::u_unary(ntt::ops::copy<TElem>{}, in_p, dim_zero, out_p,
                         dim_one, out_rest_shape.length());
        } else {
            // Non broadcast
            ntt::u_unary(ntt::ops::copy<TElem>{}, in_p, dim_one, out_p, dim_one,
                         out_rest_shape.length());
        }
    }
};

} // namespace detail

template <Tensor TIn, typename TOut>
void expand(const TIn &input, TOut &&output) noexcept {
    detail::expand_impl<TIn, std::decay_t<TOut>> impl;
    impl(input, output);
}
} // namespace nncase::ntt
