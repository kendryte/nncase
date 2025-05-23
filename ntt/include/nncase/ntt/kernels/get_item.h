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
#include "../tensor_traits.h"
#include "copy.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"

namespace nncase::ntt {
template <Tensor TIn, class TIndices, class TOut>
    requires(Dimensions<TIndices> || Dimension<TIndices>)
void get_item(const TIn &input, const TIndices &indices, TOut &&output) {
    if constexpr (Dimensions<TIndices>) {
        const auto positive_indices = positive_index(indices, input.shape());
        tensor_copy(input.view(positive_indices), output);
    } else {
        auto new_indices = make_dims(indices);
        get_item(input, new_indices, output);
    }
}
} // namespace nncase::ntt
