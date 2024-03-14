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

namespace nncase::ntt {

template <size_t Axis, typename TA, typename TB, typename TC>
void gather(const TA &input, const TB &indices, TC &&output) noexcept {
    constexpr auto rank = TA::shape_type::rank();
    constexpr auto indices_rank = TB::shape_type::rank();
    ranked_shape<rank> in_index;
    ranked_shape<indices_rank> indices_index;
    apply(output.shape(), [&](auto out_index) {
        // in_index[:axis] = out_index[:axis]
        loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });


        // in_index[axis] = indices(indices_index)
        loop<indices_rank>(
            [&](auto i) { indices_index[i] = out_index[i + Axis]; });
        in_index[Axis] = indices(indices_index);

        // in_index[axis:] = out_index[axis:]
        loop<rank - (Axis + 1)>([&](auto i) {
            in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
        });
        output(out_index) = input(in_index);
    });
}
} // namespace nncase::ntt
