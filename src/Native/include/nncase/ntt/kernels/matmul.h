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
namespace detail {
template <bool TransposedA, bool TransposedB, bool AccumulateC, class TLhs,
          class TRhs, class TOut, typename LhsPackedAxes, typename LhsPadedNums,
          typename RhsPackedAxes, typename RhsPadedNums>
class matmul_impl;

/**
 * @brief Fixed 1D-packed matmul with non transposed A/B
 * @remarks Loop orders: (k, m, n)
 */
template <bool AccumulateC, IsFixedTensor TLhs, IsFixedTensor TRhs,
          IsFixedTensor TOut, typename LhsPackedAxes, typename LhsPadedNums,
          typename RhsPackedAxes, typename RhsPadedNums>
class matmul_impl<false, false, AccumulateC, TLhs, TRhs, TOut, LhsPackedAxes,
                  LhsPadedNums, RhsPackedAxes, RhsPadedNums> {
  public:
    void operator()(const TLhs &lhs, const TRhs &rhs, TOut &output) {
        auto lhs_p = lhs.elements().data();
        auto rhs_p = rhs.elements().data();
        auto out_p = output.elements().data();
        apply<0>(lhs, rhs, output, lhs_p, rhs_p, out_p);
    }

  private:
    template <size_t Axis, class TLhsP, class TRhsP, class TOutP>
    constexpr void apply(const TLhs &lhs, const TRhs &rhs, TOut &output,
                         TLhsP lhs_p, TRhsP rhs_p, TOutP out_p) {
        // 1. Inner matmul ranks
        if constexpr (Axis == TOut::rank() - 2) {
            matmul_2d(lhs, rhs, output, lhs_p, rhs_p, out_p);
        } else {
            for (size_t i = 0; i < TOut::shape()[Axis]; i++) {
                apply<Axis + 1>(lhs, rhs, output, lhs_p, rhs_p, out_p);
                lhs_p +=
                    utility_detail::get_safe_stride(lhs, Axis, TOut::shape());
                rhs_p +=
                    utility_detail::get_safe_stride(rhs, Axis, TOut::shape());
                out_p += output.strides()[Axis];
            }
        }
    }

    template <class TLhsP, class TRhsP, class TOutP>
    constexpr void matmul_2d(const TLhs &lhs, const TRhs &rhs, TOut &output,
                             TLhsP lhs_p, TRhsP rhs_p, TOutP out_p) {
        const size_t M = output.shape()[output.rank() - 2];
        const size_t K = lhs.shape()[lhs.rank() - 1];
        const size_t N = output.shape()[output.rank() - 1];
        const size_t lhs_stride = lhs.strides()[lhs.rank() - 2];
        const size_t rhs_stride = rhs.strides()[rhs.rank() - 2];
        const size_t out_stride = output.strides()[output.rank() - 2];

        outer_product<AccumulateC>(lhs_p, rhs_p, out_p, M, N, lhs_stride,
                                   rhs_stride, out_stride);
        for (size_t k = 1; k < K; k++) {
            outer_product<true>(lhs_p, rhs_p, out_p, M, N, lhs_stride,
                                rhs_stride, out_stride);
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void outer_product(const TLhsElem *&lhs, const TRhsElem *&rhs,
                       TOutElem *output, size_t M, size_t N, size_t lhs_stride,
                       size_t rhs_stride, size_t out_stride) {
        auto lhs_mp = lhs;
        for (size_t m = 0; m < M; m++) {
            // N of B/C is always contiguous
            outer_product<AccC>(*lhs_mp, rhs, output, N);
            lhs_mp += lhs_stride;
            output += out_stride;
        }

        lhs += 1;
        rhs += rhs_stride;
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void outer_product(const TLhsElem &lhs, const TRhsElem *rhs,
                       TOutElem *output, size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            mul_add<AccC>(lhs, *rhs++, *output++);
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void mul_add(const TLhsElem &lhs, const TRhsElem &rhs, TOutElem &output) {
        // 1. 0D-packing
        if constexpr (LhsPackedAxes::rank() == 0) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2. 1D-packing
        // 2.1. pack K
        else if constexpr (LhsPackedAxes::rank() == 1 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 1) {
            static_assert(RhsPackedAxes::at(0) == TRhs::rank() - 2,
                          "B should also pack K.");
            auto value = ntt::inner_product(lhs, rhs);
            output = AccC ? output + value : value;
        } else {
            static_assert(LhsPackedAxes::rank() != 0 &&
                              LhsPackedAxes::rank() != 1,
                          "Unsupported packing.");
        }
    }
};
} // namespace detail

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
template <bool AccumulateC, class TLhs, class TRhs, class TOut,
          typename LhsPackedAxes = fixed_shape<>,
          typename LhsPadedNums = fixed_shape<>,
          typename RhsPackedAxes = fixed_shape<>,
          typename RhsPadedNums = fixed_shape<>>
void matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output,
            [[maybe_unused]] LhsPackedAxes lhsPackedAxes = {},
            [[maybe_unused]] LhsPadedNums lhsPadedNums = {},
            [[maybe_unused]] RhsPackedAxes rhsPackedAxes = {},
            [[maybe_unused]] RhsPadedNums rhsPadedNums = {}) {
    static_assert(LhsPackedAxes::rank() == RhsPackedAxes::rank(),
                  "the pack rank must equal!");
    static_assert(LhsPadedNums::rank() == RhsPadedNums::rank(),
                  "the pad rank must equal!");
    static_assert(LhsPackedAxes::rank() == 0 || LhsPackedAxes::rank() == 1,
                  "currently only support 0~1d pack!");
    static_assert(LhsPadedNums::rank() == 0 ||
                      (LhsPadedNums::at(0) == 0 && RhsPadedNums::at(0) == 0),
                  "currently only support no pad!");

    detail::matmul_impl<false, false, AccumulateC, TLhs, TRhs,
                        std::decay_t<TOut>, LhsPackedAxes, LhsPadedNums,
                        RhsPackedAxes, RhsPadedNums>
        impl;
    impl(lhs, rhs, output);
}
} // namespace nncase::ntt
