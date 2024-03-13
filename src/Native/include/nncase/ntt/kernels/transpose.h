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
#include "../utility.h"
#include <tuple>

namespace nncase::ntt {

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut>
void transpose(const TIn &input, TOut &&output) {
    constexpr auto domain = typename TIn::shape_type{};
    auto out_index = ranked_shape<domain.rank()>{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>(
            [&](auto i) { out_index[i] = index[TPerm::at(i)]; });
        output(out_index) = input(index);
    });
}
} // namespace nncase::ntt