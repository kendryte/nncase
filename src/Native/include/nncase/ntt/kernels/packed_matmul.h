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
#include "../vector_ops.h"
#include "binary.h"

namespace nncase::ntt {

namespace packed_matmul_detail {

template <class TLhs, class TRhs, class TOut, typename LhsPadedNums,
          typename RhsPadedNums>
void packed_1d_impl(const TLhs &lhs, const TRhs &rhs, TOut &&output,
                    [[maybe_unused]] LhsPadedNums lhsPadedNums,
                    [[maybe_unused]] RhsPadedNums rhsPadedNums) {
    using TElemt = typename TLhs::element_type;
    mathops::mul<TElemt> mul;
    mathops::add<TElemt> add;
    vector_ops::reduce_sum<TElemt> rvsum;
    constexpr auto lrank = TLhs::shape_type::rank();
    constexpr auto rrank = TRhs::shape_type::rank();
    auto lhs_index = ranked_shape<lrank>{};
    auto rhs_index = ranked_shape<rrank>{};
    constexpr size_t lk = lhs_index.rank() - 1;
    constexpr size_t rk = rhs_index.rank() - 2;
    apply(output.shape(), [&](auto index) {
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
        output(index) = rvsum(acc);
    });
}
} // namespace packed_matmul_detail

/**
 * @brief packed matmul
 *  have two case:
 *   1. pack 1d on the A's k and B's k
 *   2. pack 2d on the A's [m,k] and B's [k,n]
 * @param lhs
 * @param rhs
 * @param output
 * @param lhsPackedAxes
 * @param lhsPadedNums
 * @param rhsPackedAxes
 * @param rhsPadedNums
 */
template <class TLhs, class TRhs, class TOut, typename LhsPackedAxes,
          typename LhsPadedNums, typename RhsPackedAxes, typename RhsPadedNums>
void packed_matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output,
                   [[maybe_unused]] LhsPackedAxes lhsPackedAxes,
                   [[maybe_unused]] LhsPadedNums lhsPadedNums,
                   [[maybe_unused]] RhsPackedAxes rhsPackedAxes,
                   [[maybe_unused]] RhsPadedNums rhsPadedNums) {
    static_assert(LhsPackedAxes::rank() == RhsPackedAxes::rank(),
                  "the pack rank must equal!");
    static_assert(LhsPadedNums::rank() == RhsPadedNums::rank(),
                  "the pad rank must equal!");
    static_assert(LhsPackedAxes::rank() == 1,
                  "currently only support 1d pack!");
    static_assert(LhsPadedNums::rank() == 1, "currently only support 1d pack!");
    static_assert(LhsPadedNums::at(0) == 0 && RhsPadedNums::at(0) == 0,
                  "currently only support no pad!");

    if constexpr (LhsPackedAxes::rank() == 1) {
        packed_matmul_detail::packed_1d_impl(lhs, rhs, output, lhsPadedNums,
                                             rhsPadedNums);
    }
}
} // namespace nncase::ntt