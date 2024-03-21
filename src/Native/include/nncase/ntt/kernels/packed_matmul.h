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
#include "binary.h"
#include "matmul.h"

namespace nncase::ntt {

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

    if constexpr (LhsPackedAxes::rank() == 1 && RhsPackedAxes::rank() == 1) {
        if constexpr (LhsPadedNums::at(0) == 0 && RhsPadedNums::at(0) == 0) {
            matmul_detail::matmul_impl<TLhs, TRhs, std::decay_t<TOut>> impl;
            impl(lhs, rhs, output);
        }
    }
}
} // namespace nncase::ntt
