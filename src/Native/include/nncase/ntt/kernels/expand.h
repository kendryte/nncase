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
#include "../tensor_ops.h"
#include "../utility.h"

namespace nncase::ntt {

namespace expand_detail {

template <IsFixedTensor TIn, IsFixedTensor TOut>
void expand_impl(const TIn &input, TOut &&output) noexcept {
    // constexpr auto rank = TOut::shape_type::rank();
    // constexpr auto input_strides = TIn::strides();
    // constexpr auto output_strides = std::decay_t<TOut>::strides();

    using TIElem = typename TIn::element_type;
    using TOElem = typename std::decay_t<TOut>::element_type;

    static_assert(IsScalar<TOElem> && IsScalar<TIElem>,
                  "Only support scalar type for now");

    apply(output.shape(), [&](auto index) {
        const auto in_index = get_reduced_offset<input.shape().rank()>(index, input.shape());
        output(index) = input(in_index);
    });
}
} // namespace expand_detail

template <typename TIn, typename TOut>
void expand(const TIn &input, TOut &&output) noexcept {
    expand_detail::expand_impl(input, output);
}
} // namespace nncase::ntt
