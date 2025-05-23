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
#include "../shape_infer/reduce_axis.h"
#include "../tensor_traits.h"
#include <tuple>

namespace nncase::ntt {
template <Tensor... TInputs, class TOut, FixedDimension TAxis>
void concat(const std::tuple<TInputs...> &inputs, TOut &&output,
            const TAxis &axis) {
    const auto domain =
        shape_infer::reduced_shape_by_axis<TAxis::value>(output.shape());
    dynamic_shape_t<domain.rank()> in_index{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>([&](auto i) { in_index[i] = index[i]; });
        loop<sizeof...(TInputs)>([&](auto i) {
            auto input = std::get<i>(inputs);
            for (in_index[axis] = 0; in_index[axis] < input.shape()[axis];
                 in_index[axis]++, index[axis]++) {
                output(index) = input(in_index);
            }
        });
    });
}
} // namespace nncase::ntt
