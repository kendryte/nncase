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
#include "../shape_infer/reduce_axis.h"
#include "../utility.h"
#include <tuple>

namespace nncase::ntt {

template <size_t Axis, IsFixedTensor... TInputs, IsFixedTensor TOut>
void concat(const std::tuple<TInputs...> &inputs, TOut &&output) {
    constexpr auto domain = shape_infer::reduced_shape_by_axis<Axis>(
        typename std::decay_t<TOut>::shape_type{});
    auto in_index = ranked_shape<domain.rank()>{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>([&](auto i) { in_index[i] = index[i]; });
        loop<sizeof...(TInputs)>([&](auto i) {
            auto input = std::get<i>(inputs);
            for (in_index[Axis] = 0; in_index[Axis] < input.shape()[Axis];
                 in_index[Axis]++, index[Axis]++) {
                output(index) = input(in_index);
            }
        });
    });
}
} // namespace nncase::ntt