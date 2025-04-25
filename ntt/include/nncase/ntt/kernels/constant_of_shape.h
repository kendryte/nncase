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
// #include "nncase/compiler_defs.h"
#include <type_traits>

namespace nncase::ntt {

template <class TIn, class TValue, class TOut>
void constant_of_shape(const TIn &shape, const TValue &value,
                       TOut &&output) {
    using TOutType = typename std::remove_reference<TOut>::type;
    using TOutElem = typename TOutType::element_type;
    
    auto out_shape = output.shape();
    apply(out_shape, [&](auto index) {
        output(index) = (TOutElem)value(0);
    });
    // // TODO: use apply?
    // for (size_t i = 0; i < size; ++i) {
    //     output(i) = (TOutElem)value(0);
    // }
}
} // namespace nncase::ntt
