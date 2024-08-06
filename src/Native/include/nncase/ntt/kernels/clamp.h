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
#include "../loop.h"
#include "../primitive_ops.h"
#include "../tensor_ops.h"
#include "../utility.h"

namespace nncase::ntt {

namespace clamp_detail {

template <IsFixedTensor TIn, IsFixedTensor TOut, IsScalar TMin, IsScalar TMax>
void clamp_impl(const TIn &input, TOut &&output, TMin min, TMax max) {
    constexpr auto input_shape = typename TIn::shape_type{};
    apply(input_shape, [&](auto index) {
        output(index) = ntt::max(input(index), min);
        output(index) = ntt::min(output(index), max);
    });
}
} // namespace clamp_detail

template <IsFixedTensor TIn, IsFixedTensor TOut, IsScalar TMin, IsScalar TMax>
void clamp(const TIn &input, TOut &&output, TMin min, TMax max) noexcept {
    clamp_detail::clamp_impl(input, output, min, max);
}
} // namespace nncase::ntt
