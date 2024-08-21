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

namespace cast_detail {

template <IsFixedTensor TIn, IsFixedTensor TOut>
void cast_impl(const TIn &input, TOut &&output) noexcept {
    constexpr auto out_shape = std::decay_t<TOut>::shape();
    // constexpr auto rank = TOut::shape_type::rank();
    // constexpr auto input_strides = TIn::strides();
    // constexpr auto output_strides = std::decay_t<TOut>::strides();

    using element_type = typename std::decay_t<TOut>::element_type;


    apply(out_shape, [&](auto out_index) {
        if constexpr (IsTensor<element_type>) {
            apply(element_type::shape(), [&](auto index) {
                output(out_index)(index) =
                    static_cast<element_type::element_type>(
                        input(out_index)(index));
            });
        } else {
            output(out_index) = static_cast<element_type>(input(out_index));
        }
    });
}
} // namespace cast_detail

template <typename TIn, typename TOut>
void cast(const TIn &input, TOut &&output) noexcept {
    cast_detail::cast_impl(input, output);
}
} // namespace nncase::ntt
