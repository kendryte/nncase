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
template <class TDerived, Tensor TIn, Tensor TOut>
class unary_like_impl : public elementwise_impl<false, TDerived, TOut, TIn> {

  public:
    using elementwise_impl<false, TDerived, TOut, TIn>::derived;

    template <size_t Axis, class TInP, Shape TInRestShape, class TOutP,
              class... TArgs>
    constexpr void apply_contiguous(TInP &in_p, const TInRestShape &rest_shape,
                                    TOutP &out_p, TArgs &&...args) {
        const auto inner_size = rest_shape.length();
        unary_contiguous(in_p, out_p, inner_size, std::forward<TArgs>(args)...);
    }

  private:
    template <class T1, class T2, class... TArgs>
    constexpr void unary_contiguous(const T1 *input, T2 *output, size_t extent,
                                    TArgs &&...args) {
        derived().invoke_ukernel(input, output, extent,
                                 std::forward<TArgs>(args)...);
    }
};
} // namespace nncase::ntt::detail
