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
#include "../profiler.h"
#include "../utility.h"
#include <tuple>

namespace nncase::ntt {

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut>
void transpose(const TIn &input, TOut &&output) {
    constexpr auto input_shape = TIn::shape();
    constexpr auto input_strides = TIn::strides();
    constexpr auto output_shape = std::decay_t<TOut>::shape();
    constexpr auto output_strides = std::decay_t<TOut>::strides();
    constexpr auto output_rank = std::decay_t<TOut>::rank();
    constexpr auto cdims_input = contiguous_dims(input_shape, input_strides);
    constexpr auto cdims_output = contiguous_dims(output_shape, output_strides);
    if (cdims_input == TIn::rank() && cdims_output == output_rank) {
        ntt::u_transpose<TPerm>(input, output);
    } else {
        auto domain = input.shape();
        auto out_index = ranked_shape<domain.rank()>{};
        apply(domain, [&](auto index) {
            loop<domain.rank()>(
                [&](auto i) { out_index[i] = index[TPerm::at(i)]; });
            output(out_index) = input(index);
        });
    }
}

template <IsFixedDims TPerm, IsRankedTensor TIn, IsRankedTensor TOut>
void transpose(const TIn &input, TOut &&output) {
    auto domain = input.shape();
    auto out_index = ranked_shape<domain.rank()>{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>(
            [&](auto i) { out_index[i] = index[TPerm::at(i)]; });
        output(out_index) = input(index);
    });
}
} // namespace nncase::ntt