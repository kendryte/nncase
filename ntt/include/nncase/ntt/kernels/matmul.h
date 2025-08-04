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
#include "../shape_infer/matmul.h"
#include "../shape_infer/reduce.h"
#include "../ukernels.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {

template <typename T>
concept HasValidRank = requires(T t) {
    T::rank();
    requires T::rank() >= 2;
};

template <typename T>
concept ValidMatmulTensor = Tensor<T> && HasValidRank<T>;

template <class TLhs, class TRhs, typename LhsVectorizedAxes,
          typename RhsVectorizedAxes, bool TransposedA = false,
          bool TransposedB = false>
constexpr ukernels::matmul_vectorize_kind get_matmul_vectorize_kind() noexcept {
    constexpr size_t lm = TransposedA ? (TLhs::rank() - 1) : (TLhs::rank() - 2),
                     lk = TransposedA ? (TLhs::rank() - 2) : (TLhs::rank() - 1);
    constexpr size_t rk = TransposedB ? (TRhs::rank() - 1) : (TRhs::rank() - 2),
                     rn = TransposedB ? (TRhs::rank() - 2) : (TRhs::rank() - 1);

    constexpr LhsVectorizedAxes lhs_vectorized_axes;
    constexpr RhsVectorizedAxes rhs_vectorized_axes;
    if constexpr (LhsVectorizedAxes::rank() == 0 && RhsVectorizedAxes::rank() == 0) {
        return ukernels::matmul_vectorize_kind::no_vectorize;
    } else if constexpr (LhsVectorizedAxes::rank() == 1 &&
                         lhs_vectorized_axes.at(0) == lm &&
                         RhsVectorizedAxes::rank() == 0) {
        return ukernels::matmul_vectorize_kind::vectorize_m;
    } else if constexpr (LhsVectorizedAxes::rank() == 0 &&
                         RhsVectorizedAxes::rank() == 1 &&
                         rhs_vectorized_axes.at(0) == rn) {
        return ukernels::matmul_vectorize_kind::vectorize_n;
    } else if constexpr (LhsVectorizedAxes::rank() == 1 &&
                         lhs_vectorized_axes.at(0) == lk &&
                         RhsVectorizedAxes::rank() == 1 &&
                         rhs_vectorized_axes.at(0) == rk) {
        return ukernels::matmul_vectorize_kind::vectorize_k;
    } else if constexpr (LhsVectorizedAxes::rank() == 1 &&
                         lhs_vectorized_axes.at(0) == lm &&
                         RhsVectorizedAxes::rank() == 1 &&
                         rhs_vectorized_axes.at(0) == rn) {
        return ukernels::matmul_vectorize_kind::vectorize_mn;
    } else if constexpr (LhsVectorizedAxes::rank() == 2 &&
                         lhs_vectorized_axes.at(0) == lm &&
                         lhs_vectorized_axes.at(1) == lk &&
                         RhsVectorizedAxes::rank() == 1 &&
                         rhs_vectorized_axes.at(0) == rk) {
        return ukernels::matmul_vectorize_kind::vectorize_mk;
    } else if constexpr (LhsVectorizedAxes::rank() == 1 &&
                         lhs_vectorized_axes.at(0) == lk &&
                         RhsVectorizedAxes::rank() == 2 &&
                         rhs_vectorized_axes.at(0) == rk &&
                         rhs_vectorized_axes.at(1) == rn) {
        return ukernels::matmul_vectorize_kind::vectorize_kn;
    } else if constexpr (LhsVectorizedAxes::rank() == 2 &&
                         lhs_vectorized_axes.at(0) == lm &&
                         lhs_vectorized_axes.at(1) == lk &&
                         RhsVectorizedAxes::rank() == 2 &&
                         ((rhs_vectorized_axes.at(0) == rk &&
                           rhs_vectorized_axes.at(1) == rn) ||
                          (rhs_vectorized_axes.at(0) == rn &&
                           rhs_vectorized_axes.at(1) == rk))) {
        return ukernels::matmul_vectorize_kind::vectorize_mkn;
    } else {
        static_assert(TLhs::rank() == 0, "not support vectorize kind!");
        // return ukernels::matmul_vectorize_kind::unknown;
    }
}

template <bool AccumulateC, bool TransposedA, bool TransposedB, class TLhs,
          class TRhs, class TOut, typename LhsVectorizedAxes, typename LhsPadedNums,
          typename RhsVectorizedAxes, typename RhsPadedNums>
class matmul_impl;

/**
 * @brief 1D-vectorized matmul with non transposed A/B or tranposed B.
 * @remarks Loop orders: (m, n, k)
 */
template <bool AccumulateC, bool TransposedB, ValidMatmulTensor TLhs,
          ValidMatmulTensor TRhs, ValidMatmulTensor TOut,
          typename LhsVectorizedAxes, typename LhsPadedNums, typename RhsVectorizedAxes,
          typename RhsPadedNums>
class matmul_impl<AccumulateC, false, TransposedB, TLhs, TRhs, TOut,
                  LhsVectorizedAxes, LhsPadedNums, RhsVectorizedAxes, RhsPadedNums> {
    using TOutElem = typename TOut::element_type;

    static constexpr auto vectorize_kind =
        get_matmul_vectorize_kind<TLhs, TRhs, LhsVectorizedAxes, RhsVectorizedAxes, false,
                             TransposedB>();
    using policy_t =
        ntt::ukernels::u_matmul_policy<vectorize_kind, typename TLhs::value_type,
                                       typename TRhs::value_type, TOutElem,
                                       true>;
    static constexpr auto m0_subtile = policy_t::m0_subtile;

    using m1_policy_t =
        ntt::ukernels::u_matmul_m1_policy<vectorize_kind, typename TLhs::value_type,
                                          typename TRhs::value_type, TOutElem,
                                          true>;

  public:
    void operator()(const TLhs &lhs, const TRhs &rhs, TOut &output) {
        const auto domain =
            output.shape().template slice<0, TOut::rank() - 2>();
        ntt::apply(domain, [&](auto out_offset_prefix) {
            const auto out_offset = out_offset_prefix.append(0_dim, 0_dim);
            const auto lhs_offset =
                shape_infer::reduced_index_by_shape(out_offset, lhs.shape());
            const auto rhs_offset =
                shape_infer::reduced_index_by_shape(out_offset, rhs.shape());
            const auto lhs_shape = shape_infer::sub_matmul_shape(lhs.shape());
            const auto rhs_shape = shape_infer::sub_matmul_shape(rhs.shape());
            const auto out_shape =
                shape_infer::sub_matmul_shape(output.shape());

            auto a = lhs.view(lhs_offset, lhs_shape)
                         .squeeze(make_index_shape<lhs_shape.rank() - 2_dim>());
            auto b = rhs.view(rhs_offset, rhs_shape)
                         .squeeze(make_index_shape<rhs_shape.rank() - 2_dim>());
            auto c = output.view(out_offset, out_shape)
                         .squeeze(make_index_shape<out_shape.rank() - 2_dim>());
            matmul_2d_l1(a, b, c);
        });
    }

  private:
    template <class TA, class TB, class TC>
    constexpr void matmul_2d_l1(const TA &a, const TB &b, TC &c) {
        const auto M = c.shape()[c.rank() - 2_dim];
        const auto N = c.shape()[c.rank() - 1_dim];
        const auto K = a.shape()[a.rank() - 1_dim];
        constexpr auto m0_tile = policy_t::m0_tile;
        constexpr auto n0_tile = policy_t::n0_tile;

        constexpr auto m0_scale =
            ukernels::u_type_scale<vectorize_kind, TA, TB, TC>::m0_scale;
        constexpr auto n0_scale =
            ukernels::u_type_scale<vectorize_kind, TA, TB, TC>::n0_scale;

        auto scaled_M = M / m0_scale;
        auto scaled_N = N / n0_scale;

        size_t m1 = 0;
        for (; m1 < scaled_M / m0_tile * m0_tile; m1 += m0_tile) {
            size_t n1 = 0;
            for (; n1 < scaled_N / n0_tile * n0_tile; n1 += n0_tile) {
                matmul_2d_l0<m0_tile, n0_tile>(a, b, c, K, m1, n1);
            }

            if (scaled_N % n0_tile) {
                for (; n1 < scaled_N; n1++) {
                    matmul_2d_l0<m0_tile, 1>(a, b, c, K, m1, n1);
                }
            }
        }

        if (scaled_M % m0_tile) {
            for (; m1 < scaled_M; m1++) {
                constexpr auto m1_n0_tile = m1_policy_t::n0_tile;

                size_t n1 = 0;
                for (; n1 < scaled_N / m1_n0_tile * m1_n0_tile;
                     n1 += m1_n0_tile) {
                    matmul_2d_l0<1, m1_n0_tile>(a, b, c, K, m1, n1);
                }

                if (scaled_N % m1_n0_tile) {
                    for (; n1 < scaled_N; n1++) {
                        matmul_2d_l0<1, 1>(a, b, c, K, m1, n1);
                    }
                }
            }
        }
    }

    template <dim_t M0Tile, dim_t N0Tile, class TA, class TB, class TC,
              Dimension TK>
    void matmul_2d_l0(const TA &a, const TB &b, TC &c, const TK &K, dim_t m1,
                      dim_t n1) {

        constexpr auto m0_scale =
            ukernels::u_type_scale<vectorize_kind, TA, TB, TC>::m0_scale;
        constexpr auto n0_scale =
            ukernels::u_type_scale<vectorize_kind, TA, TB, TC>::n0_scale;

        constexpr auto m0_tile_scaled = m0_scale * M0Tile;
        constexpr auto n0_tile_scaled = n0_scale * N0Tile;

        auto m1_scaled = m0_scale * m1;
        auto n1_scaled = n0_scale * n1;

        auto c0 = c.view(make_shape(m1_scaled, n1_scaled),
                         fixed_shape_v<m0_tile_scaled, n0_tile_scaled>);
        auto a1 =
            a.view(make_shape(m1, 0_dim), make_shape(fixed_dim_v<M0Tile>, K));
        auto b1 =
            b.view(ntt::where(std::integral_constant<bool, TransposedB>{},
                              make_shape(n1, 0_dim), make_shape(0_dim, n1)),
                   ntt::where(std::integral_constant<bool, TransposedB>{},
                              make_shape(fixed_dim_v<N0Tile>, K),
                              make_shape(K, fixed_dim_v<N0Tile>)));
        ntt::u_matmul<vectorize_kind, AccumulateC, false, TransposedB, M0Tile,
                      N0Tile>(a1, b1, c0, K);
    }
};
} // namespace detail

/**
 * @brief vectorized matmul
 *  have two case:
 *   1. vectorize 1d on the A's k and B's k
 *   2. vectorize 2d on the A's [m,k] and B's [k,n]
 * @param lhs
 * @param rhs
 * @param output
 * @param lhsVectorizedAxes
 * @param lhsPadedNums
 * @param rhsVectorizedAxes
 * @param rhsPadedNums
 */
template <bool AccumulateC = false, bool TransposedA = false,
          bool TransposedB = false, Tensor TLhs, Tensor TRhs, class TOut,
          FixedDimensions LhsVectorizedAxes = shape_t<>,
          FixedDimensions LhsPadedNums = shape_t<>,
          FixedDimensions RhsVectorizedAxes = shape_t<>,
          FixedDimensions RhsPadedNums = shape_t<>>
void matmul(
    [[maybe_unused]] const TLhs &lhs, [[maybe_unused]] const TRhs &rhs,
    [[maybe_unused]] TOut &&output,
    [[maybe_unused]] const LhsVectorizedAxes &lhsVectorizedAxes = fixed_shape_v<>,
    [[maybe_unused]] const LhsPadedNums &lhsPadedNums = fixed_shape_v<>,
    [[maybe_unused]] const RhsVectorizedAxes &rhsVectorizedAxes = fixed_shape_v<>,
    [[maybe_unused]] const RhsPadedNums &rhsPadedNums = fixed_shape_v<>) {

    constexpr LhsPadedNums lhsPadedNumsType;
    constexpr RhsPadedNums rhsPadedNumsType;
    static_assert(LhsVectorizedAxes::rank() == 0 || LhsVectorizedAxes::rank() == 1 ||
                      LhsVectorizedAxes::rank() == 2,
                  "currently only support 0~2d vectorize!");
    static_assert(RhsVectorizedAxes::rank() == 0 || RhsVectorizedAxes::rank() == 1 ||
                      RhsVectorizedAxes::rank() == 2,
                  "currently only support 0~2d vectorize!");
    static_assert(LhsPadedNums::rank() == 0 || lhsPadedNumsType.length() == 0,
                  "currently only support no pad!");
    static_assert(RhsPadedNums::rank() == 0 || rhsPadedNumsType.length() == 0,
                  "currently only support no pad!");

    detail::matmul_impl<AccumulateC, TransposedA, TransposedB, TLhs, TRhs,
                        std::decay_t<TOut>, LhsVectorizedAxes, LhsPadedNums,
                        RhsVectorizedAxes, RhsPadedNums>
        impl;
    impl(lhs, rhs, output);

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
    // TODO: remove this when tiling is ready
    using TLhsElem = typename TLhs::element_type;
    using TRhsElem = typename TRhs::element_type;
    using TOutElem = typename std::decay_t<TOut>::element_type;
    if constexpr (Vector<TLhsElem> && Vector<TRhsElem>) {
        if constexpr (TLhsElem::shape_type::rank() == 2 &&
                      TRhsElem::shape_type::rank() == 2 &&
                      std::is_same_v<typename TOutElem::shape_type,
                                     fixed_shape_t<64, 64>>) {

            ntt::apply(output.shape(), [&](auto index) {
                auto data = (float *)output(index).buffer().data();
                float tmp[64 * 64];
                for (int i = 0; i < 64; i++) {
                    std::memcpy(tmp + i * 2 * 32, data + i * 32,
                                32 * sizeof(float));
                    std::memcpy(tmp + (i * 2 + 1) * 32, data + (i + 64) * 32,
                                32 * sizeof(float));
                }

                std::memcpy(data, tmp, 64 * 64 * sizeof(float));
            });
        }
    }
#endif
}
} // namespace nncase::ntt
