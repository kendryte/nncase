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
#include "../utility.h"

namespace nncase::ntt {

namespace pad_detail {

template <IsFixedTensor TIn, IsFixedTensor TOut, typename TElem, size_t... Ints>
void pad_impl(const TIn &input, TOut &&output, const TElem &padValue,
              const fixed_shape<Ints...> paddings) {
    constexpr auto input_shape = TIn::shape();
    constexpr auto rank = TIn::shape().rank();
    static_assert(sizeof...(Ints) == rank * 2, "the paddings not support!");
    constexpr auto output_shape = std::decay_t<TOut>::shape();
    // constexpr auto input_strides = TIn::strides();
    // constexpr auto output_strides = std::decay_t<TOut>::strides();
    auto in_index = ranked_shape<rank>();
    apply(output_shape, [&](auto out_index) {
        bool dopad = false;
        for (size_t i = 0; i < rank; i++) {
            in_index[i] = out_index[i] - paddings.at(i * 2);
            if (in_index[i] < 0 || in_index[i] >= input_shape.at(i)) {
                dopad = true;
                break;
            }
        }
        if (dopad) {
            output(out_index) = padValue;
        } else {
            output(out_index) = input(in_index);
        }
    });
}
} // namespace pad_detail

/**
 * @brief pad
 * 
 * @tparam Paddings   (dim 0 before, after, dim 1 before, after,...)
 * @param input input tensor.
 * @param output output tensor.
 * @param padValue pad value.
 */
template <size_t... Paddings, typename TIn, typename TOut, typename TElem>
void pad(const TIn &input, TOut &&output, const TElem &padValue) noexcept {
    pad_detail::pad_impl(input, output, padValue, fixed_shape<Paddings...>{});
}
} // namespace nncase::ntt
