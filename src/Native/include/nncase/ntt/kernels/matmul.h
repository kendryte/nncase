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
#include "binary.h"

namespace nncase::ntt {
template <class TLhs, class TRhs, class TOut>
void matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    auto out_shape = output.shape();
    using TElemt = typename TLhs::element_type;
    mathops::mul<TElemt> mul;
    mathops::add<TElemt> add;
    apply(out_shape, [&](auto index) {
        constexpr auto lrank = TLhs::shape_type::rank();
        constexpr auto rrank = TRhs::shape_type::rank();
        auto lhs_index = ranked_shape<lrank>{};
        auto rhs_index = ranked_shape<rrank>{};
        constexpr size_t lk = lhs_index.rank() - 1;
        constexpr size_t rk = rhs_index.rank() - 2;
        for (size_t i = 0; i < lk; i++) {
            lhs_index[i] = index[i];
        }
        for (size_t i = 0; i < rk; i++) {
            rhs_index[i] = index[i];
        }
        rhs_index[rk + 1] = index[rk + 1];
        TElemt acc = 0;
        for (lhs_index[lk] = 0; lhs_index[lk] < lhs.shape()[lk];
             lhs_index[lk]++) {
            rhs_index[rk] = lhs_index[lk];
            TElemt val = mul(lhs(lhs_index), rhs(rhs_index));
            acc = add(acc, val);
        }
        output(index) = acc;
    });
}
} // namespace nncase::ntt
