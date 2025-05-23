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
#include "../ukernels.h"

namespace nncase::ntt {
template <Tensor TIn, Tensor TOut, FixedDimensions TPerm>
    requires(TIn::rank() == TOut::rank() && TIn::rank() == TPerm::rank())
void transpose(const TIn &input, TOut &&output,
               const TPerm &perm = make_index_shape<TIn::rank()>().reverse()) {
    auto domain = input.shape();
    dynamic_shape_t<domain.rank()> out_index{};
    apply(domain, [&](auto index) {
        loop<domain.rank()>(
            [&](auto i) { out_index[i] = index[perm.template at<i>()]; });
        output(out_index) = input(index);
    });
}
} // namespace nncase::ntt
