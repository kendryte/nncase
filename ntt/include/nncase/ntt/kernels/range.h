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
#include "nncase/compiler_defs.h"
#include <type_traits>

namespace nncase::ntt {

template <class TBegin, class TEnd, class TStep, class TOut>
void range(const TBegin &begin, NNCASE_UNUSED const TEnd &end,
           const TStep &step, TOut &&output) {
    // FIXME: rewrite this with a better way
    using element_type = element_or_scalar_t<TOut>;
    constexpr auto output_shape = typename TOut::shape_type{};
    apply(output_shape, [&](auto shape) {
        element_type value = begin + step * shape;
        output(shape) = value;
    });
}
} // namespace nncase::ntt
