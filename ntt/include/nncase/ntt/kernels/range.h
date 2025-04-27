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

template <class TBegin, class TEnd, class TStep, class TOut>
void range(const TBegin &begin, const TEnd &end, const TStep &step, TOut &&output) {
    using TOutType = typename std::remove_reference<TOut>::type;
    using TOutElem = typename TOutType::element_type;
    const auto size = output.size();

    for (size_t i = 0; i < size; ++i) {
        output(i) = (TOutElem)begin(0) + (TOutElem)step(0) * (TOutElem)i;
    }
}
} // namespace nncase::ntt
