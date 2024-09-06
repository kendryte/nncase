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
#include "../profiler.h"
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

        outer_product<AccumulateC>(lhs_p, rhs_p, out_p, M, N, K, lhs_stride,
                                   rhs_stride, out_stride);

        if constexpr (LhsPackedAxes::rank() == 1 &&
                      LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                      RhsPackedAxes::rank() == 1 &&
                      RhsPackedAxes::at(0) == TRhs::rank() - 2) {
            return;
        }

        if constexpr (LhsPackedAxes::rank() == 2 &&
                      LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                      LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                      RhsPackedAxes::rank() == 1 &&
                      RhsPackedAxes::at(0) == TRhs::rank() - 2) {
            return;
        }

        for (size_t k = 1; k < K; k++) {
            outer_product<true>(lhs_p, rhs_p, out_p, M, N, K, lhs_stride,
                                rhs_stride, out_stride);
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void outer_product(const TLhsElem *&lhs, const TRhsElem *&rhs,
                       TOutElem *output, size_t M, size_t N,
                       [[maybe_unused]] size_t K, size_t lhs_stride,
                       size_t rhs_stride, size_t out_stride) {

        // 1. 1D-packing: pack K
        if constexpr (LhsPackedAxes::rank() == 1 &&
                      LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                      RhsPackedAxes::rank() == 1 &&
                      RhsPackedAxes::at(0) == TRhs::rank() - 2) {
            auto lhs_mp = lhs;
            for (size_t m = 0; m < M; m++) {
                outer_product<AccC>(lhs_mp, rhs, output, N, K, rhs_stride);
                lhs_mp += lhs_stride;
                output += out_stride;
            }
        }
        // 2. 2D-packing: pack MK & K
        else if constexpr (LhsPackedAxes::rank() == 2 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                           LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                           RhsPackedAxes::rank() == 1 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 2) {
            auto lhs_mp = lhs;
            for (size_t m = 0; m < M; m++) {
                outer_product<AccC>(lhs_mp, rhs, output, N, K, rhs_stride);
                lhs_mp += lhs_stride;
                output += out_stride;
            }
        }
        // 3. Other Case
        else {
            auto lhs_mp = lhs;
            for (size_t m = 0; m < M; m++) {
                outer_product<AccC>(*lhs_mp, rhs, output, N);
                lhs_mp += lhs_stride;
                output += out_stride;
            }
            lhs += 1;
            rhs += rhs_stride;
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void outer_product(const TLhsElem &lhs, const TRhsElem *rhs,
                       TOutElem *output, size_t extent) {
        for (size_t i = 0; i < extent; i++) {
            mul_add<AccC>(lhs, *rhs++, *output++);
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    requires(TLhsElem::rank() ==
             1) void outer_product(const TLhsElem *lhs, const TRhsElem *rhs,
                                   TOutElem *output, size_t extent, size_t K,
                                   size_t rhs_stride) {
        for (size_t i = 0; i < extent; i++) {
            auto rhs_mp = rhs;
            auto lhs_mp = lhs;
            for (size_t k = 0; k < K; k++) {
                auto value = ntt::inner_product(*lhs_mp, *rhs_mp);
                *output = AccumulateC || k > 0 ? *output + value : value;
                lhs_mp++;
                rhs_mp += rhs_stride;
            }
            rhs++;
            output++;
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    requires(TLhsElem::rank() ==
             2) void outer_product(const TLhsElem *lhs, const TRhsElem *rhs,
                                   TOutElem *output, size_t extent, size_t K,
                                   size_t rhs_stride) {
        for (size_t i = 0; i < extent; i++) {
            auto rhs_mp = rhs;
            auto lhs_mp = lhs;
            for (size_t k = 0; k < K; k++) {
                for (size_t m = 0; m < TLhsElem::shape().at(0); m++) {
                    auto value = ntt::inner_product((*lhs_mp)(m), *rhs_mp);
                    (*output)(m) =
                        AccumulateC || k > 0 ? (*output)(m) + value : value;
                }
                lhs_mp++;
                rhs_mp += rhs_stride;
            }
            rhs++;
            output++;
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void mul_add(const TLhsElem &lhs, const TRhsElem &rhs, TOutElem &output) {
        // 1. 0D-packing
        if constexpr (LhsPackedAxes::rank() == 0 &&
                      RhsPackedAxes::rank() == 0) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2. 1D-packing
        // 2.1. pack M
        else if constexpr (LhsPackedAxes::rank() == 1 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                           RhsPackedAxes::rank() == 0) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2.2. pack N
        else if constexpr (LhsPackedAxes::rank() == 0 &&
                           RhsPackedAxes::rank() == 1 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 1) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2.3. pack M & N
        else if constexpr (LhsPackedAxes::rank() == 1 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                           RhsPackedAxes::rank() == 1 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 1) {
            auto value = ntt::outer_product(lhs, rhs);
            output = AccC ? output + value : value;
        }
        // 3.2. pack K & KN
        else if constexpr (LhsPackedAxes::rank() == 1 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                           RhsPackedAxes::rank() == 2 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 2 &&
                           RhsPackedAxes::at(1) == TRhs::rank() - 1) {
            fixed_tensor_alike_t<TLhsElem, 1, TLhsElem::shape().at(0)> lhs_2d{
                {lhs}};
            fixed_tensor_alike_t<TOutElem, 1, TOutElem::shape().at(0)>
                output_2d{{output}};
            output_2d = ntt::mma<AccC>(lhs_2d, rhs, output_2d);
            output = output_2d(0);
        }
        // 3.3. pack MK & KN
        else if constexpr (LhsPackedAxes::rank() == 2 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                           LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                           RhsPackedAxes::rank() == 2 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 2 &&
                           RhsPackedAxes::at(1) == TRhs::rank() - 1) {
            output = ntt::mma<AccC>(lhs, rhs, output);
        } else {
            static_assert(sizeof(TLhsElem) == 0, "Unsupported packing.");
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
    static_assert(LhsPackedAxes::rank() == 0 || LhsPackedAxes::rank() == 1 ||
                      LhsPackedAxes::rank() == 2,
                  "currently only support 0~2d pack!");
    static_assert(RhsPackedAxes::rank() == 0 || RhsPackedAxes::rank() == 1 ||
                      RhsPackedAxes::rank() == 2,
                  "currently only support 0~2d pack!");
    static_assert(LhsPadedNums::rank() == 0 || LhsPadedNums::length() == 0,
                  "currently only support no pad!");
    static_assert(RhsPadedNums::rank() == 0 || RhsPadedNums::length() == 0,
                  "currently only support no pad!");

    AUTO_NTT_PROFILER

    detail::matmul_impl<false, false, AccumulateC, TLhs, TRhs,
                        std::decay_t<TOut>, LhsPackedAxes, LhsPadedNums,
                        RhsPackedAxes, RhsPadedNums>
        impl;
    impl(lhs, rhs, output);
}
} // namespace nncase::ntt
