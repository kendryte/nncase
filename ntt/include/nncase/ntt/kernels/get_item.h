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
#include "../tensor_ops.h"
#include "../tensor_traits.h"
#include "../ukernels.h"
#include "../utility.h"
#include "copy.h"
#include <type_traits>

namespace nncase::ntt {

template <class TIn, class TIndices, class TOut>
void get_item(const TIn &input, const TIndices &indices, TOut &&output) {
    constexpr size_t indices_rank = TIndices::rank();
    ranked_shape<indices_rank> indices_dims{};
    if constexpr (indices_rank) {
        for (size_t i = 0; i < indices_dims.rank(); i++) {
            indices_dims[i] = positive_index(indices[i], input.shape()[i]);
        }
    } else {
        indices_dims[0] = indices(fixed_shape<0>{});
    }
    tensor_copy(input.view(indices_dims), output);
}
} // namespace nncase::ntt
