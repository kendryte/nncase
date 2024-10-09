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
#include "../shape_infer/matmul.h"
#include "../ukernels.h"
#include "nncase/ntt/primitive_ops.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/shape_infer/reduce.h"
#include "nncase/ntt/utility.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TLhs, class TRhs, typename LhsPackedAxes,
          typename RhsPackedAxes, bool TransA = false, bool TransB = false>
constexpr ukernels::mamtul_pack_kind get_matmul_pack_kind() noexcept {
    constexpr size_t lm = TransA ? (TLhs::rank() - 1) : (TLhs::rank() - 2),
                     lk = TransA ? (TLhs::rank() - 2) : (TLhs::rank() - 1);
    constexpr size_t rk = TransB ? (TRhs::rank() - 1) : (TRhs::rank() - 2),
                     rn = TransB ? (TRhs::rank() - 2) : (TRhs::rank() - 1);
    if constexpr (LhsPackedAxes::rank() == 0 && RhsPackedAxes::rank() == 0) {
        return ukernels::mamtul_pack_kind::no_pack;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == lm &&
                         RhsPackedAxes::rank() == 0) {
        return ukernels::mamtul_pack_kind::pack_m;
    } else if constexpr (LhsPackedAxes::rank() == 0 &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == rn) {
        return ukernels::mamtul_pack_kind::pack_n;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == lk &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == rk) {
        return ukernels::mamtul_pack_kind::pack_k;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == lm &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == rn) {
        return ukernels::mamtul_pack_kind::pack_mn;
    } else if constexpr (LhsPackedAxes::rank() == 2 &&
                         LhsPackedAxes::at(0) == lm &&
                         LhsPackedAxes::at(1) == lk &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == rk) {
        return ukernels::mamtul_pack_kind::pack_mk;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == lk &&
                         RhsPackedAxes::rank() == 2 &&
                         RhsPackedAxes::at(0) == rk &&
                         RhsPackedAxes::at(1) == rn) {
        return ukernels::mamtul_pack_kind::pack_kn;
    } else if constexpr (LhsPackedAxes::rank() == 2 &&
                         LhsPackedAxes::at(0) == lm &&
                         LhsPackedAxes::at(1) == lk &&
                         RhsPackedAxes::rank() == 2 &&
                         RhsPackedAxes::at(0) == rk &&
                         RhsPackedAxes::at(1) == rn) {
        return ukernels::mamtul_pack_kind::pack_mkn;
    } else {
        return ukernels::mamtul_pack_kind::unknown;
    }
}

template <bool TransposedA, bool TransposedB, bool AccumulateC, class TLhs,
          class TRhs, class TOut, typename LhsPackedAxes, typename LhsPadedNums,
          typename RhsPackedAxes, typename RhsPadedNums>
class matmul_impl;

/**
 * @brief fixed matmul with transpose B
 * @remark loop order is m,n,k
 */
template <bool AccumulateC, class TLhs, class TRhs, class TOut,
          typename LhsPackedAxes, typename LhsPadedNums, typename RhsPackedAxes,
          typename RhsPadedNums>
class matmul_impl<false, true, AccumulateC, TLhs, TRhs, TOut, LhsPackedAxes,
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

        for (size_t m = 0; m < M; m++) {
            auto out_p_inner = out_p;
            for (size_t n = 0; n < N; n++) {
                inner_product<AccumulateC>(lhs_p + (m * lhs_stride),
                                           rhs_p + (n * rhs_stride),
                                           *out_p_inner++, K);
            }
            out_p += out_stride;
        }
    }
    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void inner_product(const TLhsElem *lhs, const TRhsElem *rhs,
                       TOutElem &output, const size_t K) {
        // 2.4. pack M & N
        if constexpr (LhsPackedAxes::rank() == 1 &&
                      LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                      RhsPackedAxes::rank() == 1 &&
                      RhsPackedAxes::at(0) == TRhs::rank() - 2) {
            auto value = ntt::outer_product(*lhs++, *rhs++);
            output = AccC ? output + value : value;
            for (size_t k = 1; k < K; k++) {
                output += ntt::outer_product(*lhs++, *rhs++);
            }
        }
        // 3.3. pack [M,K]<m,k> & [N,K]<k,n>
        else if constexpr (LhsPackedAxes::rank() == 2 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                           LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                           RhsPackedAxes::rank() == 2 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 1 &&
                           RhsPackedAxes::at(1) == TRhs::rank() - 2) {
            output = ntt::mma<AccC, false>(*lhs++, *rhs++, output);
            for (size_t k = 1; k < K; k++) {
                output = ntt::mma<true, false>(*lhs++, *rhs++, output);
            }
        }
        // 3.3. pack [M,K]<k,m> & [N,K]<k,n>
        else if constexpr (LhsPackedAxes::rank() == 2 &&
                           LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                           LhsPackedAxes::at(1) == TLhs::rank() - 2 &&
                           RhsPackedAxes::rank() == 2 &&
                           RhsPackedAxes::at(0) == TRhs::rank() - 1 &&
                           RhsPackedAxes::at(1) == TRhs::rank() - 2) {
            output = ntt::mma<AccC, true>(*lhs++, *rhs++, output);
            for (size_t k = 1; k < K; k++) {
                output = ntt::mma<true, true>(*lhs++, *rhs++, output);
            }
        }
        // fall back
        else {
            output = AccC ? ntt::mul_add(*lhs++, *rhs++, output)
                          : ntt::mul(*lhs++, *rhs++);
            for (size_t k = 1; k < K; k++) {
                output = ntt::mul_add(*lhs++, *rhs++, output);
            }
        }
    }
};

/**
 * @brief Fixed 1D-packed matmul with non transposed A/B
 * @remarks Loop orders: (m, n, k)
 */
template <bool AccumulateC, IsFixedTensor TLhs, IsFixedTensor TRhs,
          IsFixedTensor TOut, typename LhsPackedAxes, typename LhsPadedNums,
          typename RhsPackedAxes, typename RhsPadedNums>
class matmul_impl<false, false, AccumulateC, TLhs, TRhs, TOut, LhsPackedAxes,
                  LhsPadedNums, RhsPackedAxes, RhsPadedNums> {
    using TOutElem = typename TOut::element_type;

    static constexpr auto pack_kind =
        get_matmul_pack_kind<TLhs, TRhs, LhsPackedAxes, RhsPackedAxes>();
    using policy_t =
        ntt::ukernels::u_matmul_policy<pack_kind, typename TLhs::element_type,
                                       typename TRhs::element_type, TOutElem,
                                       true>;
    static constexpr auto m0_subtile = policy_t::m0_subtile;

  public:
    void operator()(const TLhs &lhs, const TRhs &rhs, TOut &output) {
        auto domain =
            slice_fixed_dims<TOut::rank() - 2>(typename TOut::shape_type{});
        ntt::apply(domain, [&](auto out_offset_prefix) {
            ranked_shape<TOut::rank()> out_offset{};
            std::copy(out_offset_prefix.begin(), out_offset_prefix.end(),
                      out_offset.begin());
            auto lhs_offset =
                shape_infer::reduced_index_by_shape(out_offset, TLhs::shape());
            auto rhs_offset =
                shape_infer::reduced_index_by_shape(out_offset, TRhs::shape());
            auto lhs_shape = shape_infer::sub_matmul_shape(TLhs::shape());
            auto rhs_shape = shape_infer::sub_matmul_shape(TRhs::shape());
            auto out_shape = shape_infer::sub_matmul_shape(TOut::shape());

            auto a = lhs.view(lhs_offset, lhs_shape)
                         .squeeze(make_index_axes<lhs_shape.rank() - 2>());
            auto b = rhs.view(rhs_offset, rhs_shape)
                         .squeeze(make_index_axes<rhs_shape.rank() - 2>());
            auto c = output.view(out_offset, out_shape)
                         .squeeze(make_index_axes<out_shape.rank() - 2>());
            matmul_2d_l1(a, b, c);
        });
    }

  private:
    template <class TA, class TB, class TC>
    constexpr void matmul_2d_l1(const TA &a, const TB &b, TC &c) {
        size_t M = c.shape()[c.rank() - 2];
        size_t N = c.shape()[c.rank() - 1];
        size_t K = a.shape()[a.rank() - 1];
        constexpr auto m0_tile = policy_t::m0_tile;
        constexpr auto n0_tile = policy_t::n0_tile;

        size_t m1 = 0;
        for (; m1 < M / m0_tile * m0_tile; m1 += m0_tile) {
            size_t n1 = 0;
            for (; n1 < N / n0_tile * n0_tile; n1 += n0_tile) {
                matmul_2d_l0<m0_tile, n0_tile>(a, b, c, K, m1, n1);
            }

            if (N % n0_tile) {
                for (; n1 < N; n1++) {
                    matmul_2d_l0<m0_tile, 1>(a, b, c, K, m1, n1);
                }
            }
        }

        if (M % m0_tile) {
            for (; m1 < M; m1++) {
                size_t n1 = 0;
                for (; n1 < N / n0_tile * n0_tile; n1 += n0_tile) {
                    matmul_2d_l0<1, n0_tile>(a, b, c, K, m1, n1);
                }

                if (N % n0_tile) {
                    for (; n1 < N; n1++) {
                        matmul_2d_l0<1, 1>(a, b, c, K, m1, n1);
                    }
                }
            }
        }
    }

    template <size_t M0Tile, size_t N0Tile, class TA, class TB, class TC>
    void matmul_2d_l0(const TA &a, const TB &b, TC &c, size_t K, size_t m1,
                      size_t n1) {
        auto c0 =
            c.view(make_ranked_shape(m1, n1), fixed_shape<M0Tile, N0Tile>{});
        auto a1 =
            a.view(make_ranked_shape(m1, 0), make_ranked_shape(M0Tile, K));
        auto b1 =
            b.view(make_ranked_shape(0, n1), make_ranked_shape(K, N0Tile));

        ntt::u_matmul<pack_kind, AccumulateC, M0Tile, N0Tile>(a1, b1, c0, K);
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

template <bool AccumulateC, bool TransposeA, bool TransposeB, class TLhs,
          class TRhs, class TOut, typename LhsPackedAxes = fixed_shape<>,
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

    detail::matmul_impl<TransposeA, TransposeB, AccumulateC, TLhs, TRhs,
                        std::decay_t<TOut>, LhsPackedAxes, LhsPadedNums,
                        RhsPackedAxes, RhsPadedNums>
        impl;
    impl(lhs, rhs, output);
}
} // namespace nncase::ntt
