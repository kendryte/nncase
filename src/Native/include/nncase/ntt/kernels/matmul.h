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
          typename RhsPackedAxes>
constexpr ukernels::mamtul_pack_kind get_matmul_pack_kind() noexcept {
    if constexpr (LhsPackedAxes::rank() == 0 && RhsPackedAxes::rank() == 0) {
        return ukernels::mamtul_pack_kind::no_pack;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                         RhsPackedAxes::rank() == 0) {
        return ukernels::mamtul_pack_kind::pack_m;
    } else if constexpr (LhsPackedAxes::rank() == 0 &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 1) {
        return ukernels::mamtul_pack_kind::pack_n;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 2) {
        return ukernels::mamtul_pack_kind::pack_k;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 1) {
        return ukernels::mamtul_pack_kind::pack_mn;
    } else if constexpr (LhsPackedAxes::rank() == 2 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                         LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                         RhsPackedAxes::rank() == 1 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 2) {
        return ukernels::mamtul_pack_kind::pack_mk;
    } else if constexpr (LhsPackedAxes::rank() == 1 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 1 &&
                         RhsPackedAxes::rank() == 2 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 2 &&
                         RhsPackedAxes::at(1) == TRhs::rank() - 1) {
        return ukernels::mamtul_pack_kind::pack_kn;
    } else if constexpr (LhsPackedAxes::rank() == 2 &&
                         LhsPackedAxes::at(0) == TLhs::rank() - 2 &&
                         LhsPackedAxes::at(1) == TLhs::rank() - 1 &&
                         RhsPackedAxes::rank() == 2 &&
                         RhsPackedAxes::at(0) == TRhs::rank() - 2 &&
                         RhsPackedAxes::at(1) == TRhs::rank() - 1) {
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

        // 1. pack M & N
        if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mn &&
                      m0_subtile) {
            using TSubOutElem = ntt::vector<typename TOutElem::element_type,
                                            TOutElem::shape().last()>;
            TSubOutElem c0_tmp[m0_subtile][N0Tile];

            for (size_t sm1 = 0; sm1 < TOutElem::shape()[0];
                 sm1 += m0_subtile) {
                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0_tmp[index[0]][index[1]] =
                        AccumulateC ? c0(0, index[1])(sm1 + index[0])
                                    : TSubOutElem{};
                });

                for (size_t k1 = 0; k1 < K; k1++) {
                    outer_product<M0Tile, N0Tile>(a, b, c0_tmp, m1, k1, n1,
                                                  sm1);
                }

                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0(0, index[1])(sm1 + index[0]) =
                        c0_tmp[index[0]][index[1]];
                });
            }
        }
        // 2. pack K & KN
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_kn) {
            using TLhsElem = std::remove_const_t<typename TA::element_type>;

            TOutElem c0_tmp[M0Tile][N0Tile];
            ntt::apply(c0.shape(), [&](auto index) {
                c0_tmp[index[0]][index[1]] =
                    AccumulateC ? c0(index) : TOutElem{};
            });

            for (size_t k1 = 0; k1 < K; k1++) {
                for (size_t sk1 = 0; sk1 < TLhsElem::shape()[0]; sk1++) {
                    outer_product<M0Tile, N0Tile>(a, b, c0_tmp, m1, k1, n1, 0,
                                                  sk1);
                }
            }

            ntt::apply(c0.shape(), [&](auto index) {
                c0(index) = c0_tmp[index[0]][index[1]];
            });
        }
        // 3. pack MK & KN
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mkn &&
                           m0_subtile) {
            using TLhsElem = std::remove_const_t<typename TA::element_type>;
            using TSubOutElem = ntt::vector<typename TOutElem::element_type,
                                            TOutElem::shape().last()>;

            TSubOutElem c0_tmp[m0_subtile][N0Tile];

            for (size_t sm1 = 0; sm1 < TOutElem::shape()[0];
                 sm1 += m0_subtile) {
                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0_tmp[index[0]][index[1]] =
                        AccumulateC ? c0(0, index[1])(sm1 + index[0])
                                    : TSubOutElem{};
                });

                for (size_t k1 = 0; k1 < K; k1++) {
                    for (size_t sk1 = 0; sk1 < TLhsElem::shape()[0]; sk1++) {
                        outer_product<M0Tile, N0Tile>(a, b, c0_tmp, m1, k1, n1,
                                                      sm1, sk1);
                    }
                }

                ntt::apply(fixed_shape<m0_subtile, N0Tile>{}, [&](auto index) {
                    c0(0, index[1])(sm1 + index[0]) =
                        c0_tmp[index[0]][index[1]];
                });
            }
        }
        // Other packs
        else {
            TOutElem c0_tmp[M0Tile][N0Tile];
            ntt::apply(c0.shape(), [&](auto index) {
                c0_tmp[index[0]][index[1]] =
                    AccumulateC ? c0(index) : TOutElem{};
            });

            for (size_t k1 = 0; k1 < K; k1++) {
                outer_product<M0Tile, N0Tile>(a, b, c0_tmp, m1, k1, n1);
            }

            ntt::apply(c0.shape(), [&](auto index) {
                c0(index) = c0_tmp[index[0]][index[1]];
            });
        }
    }

    template <size_t M0Tile, size_t N0Tile, class TA, class TB, class TC>
    void outer_product(const TA &a, const TB &b, TC &c0_tmp, size_t m1,
                       size_t k1, size_t n1, size_t sm1 = 0, size_t sk1 = 0) {
        auto a1 =
            a.view(make_ranked_shape(m1, k1), make_ranked_shape(M0Tile, 1));
        auto b1 =
            b.view(make_ranked_shape(k1, n1), make_ranked_shape(1, N0Tile));

        using TLhsElem = std::remove_const_t<typename TA::element_type>;
        using TRhsElem = std::remove_const_t<typename TB::element_type>;

        // 1. pack M & N
        if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mn &&
                      m0_subtile) {
            using TSubLhsElem = typename TLhsElem::element_type;
            TSubLhsElem a0_tmp[m0_subtile];
            TRhsElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape<m0_subtile>{}, [&](auto index) {
                a0_tmp[index[0]] = a1(0, 0)(sm1 + index[0]);
            });
            ntt::apply(fixed_shape<N0Tile>{},
                       [&](auto index) { b0_tmp[index[0]] = b1(0, index[0]); });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < m0_subtile; m++) {
                    mul_add<true>(a0_tmp[m], b0_tmp[n], c0_tmp[m][n]);
                }
            }
        }
        // 2. pack K & KN
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_kn) {
            using TSubLhsElem = typename TLhsElem::element_type;
            using TSubRhsElem = ntt::vector<typename TRhsElem::element_type,
                                            TRhsElem::shape().last()>;
            TSubLhsElem a0_tmp[M0Tile];
            TSubRhsElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape<M0Tile>{}, [&](auto index) {
                a0_tmp[index[0]] = a1(index[0], 0)(sk1);
            });
            ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                b0_tmp[index[0]] = b1(0, index[0])(sk1);
            });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < M0Tile; m++) {
                    mul_add<true>(a0_tmp[m], b0_tmp[n], c0_tmp[m][n]);
                }
            }
        }
        // 1. pack MK & KN
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mkn &&
                           m0_subtile) {
            using TSubLhsElem = typename TLhsElem::element_type;
            using TSubRhsElem = ntt::vector<typename TRhsElem::element_type,
                                            TRhsElem::shape().last()>;
            TSubLhsElem a0_tmp[m0_subtile];
            TSubRhsElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape<m0_subtile>{}, [&](auto index) {
                a0_tmp[index[0]] = a1(0, 0)(sm1 + index[0], sk1);
            });
            ntt::apply(fixed_shape<N0Tile>{}, [&](auto index) {
                b0_tmp[index[0]] = b1(0, index[0])(sk1);
            });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < m0_subtile; m++) {
                    auto &output = c0_tmp[m][n];
                    auto value = ntt::outer_product(a0_tmp[m], b0_tmp[n]);
                    output = output + value;
                }
            }
        }
        // Other packs
        else {
            TLhsElem a0_tmp[M0Tile];
            TRhsElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape<M0Tile>{},
                       [&](auto index) { a0_tmp[index[0]] = a1(index[0], 0); });
            ntt::apply(fixed_shape<N0Tile>{},
                       [&](auto index) { b0_tmp[index[0]] = b1(0, index[0]); });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < M0Tile; m++) {
                    mul_add<true>(a0_tmp[m], b0_tmp[n], c0_tmp[m][n]);
                }
            }
        }
    }

    template <bool AccC, class TLhsElem, class TRhsElem, class TOutElem>
    void mul_add(const TLhsElem &lhs, const TRhsElem &rhs, TOutElem &output) {
        // 1. 0D-packing
        if constexpr (pack_kind == ukernels::mamtul_pack_kind::no_pack) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2. 1D-packing
        // 2.1. pack M
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_m) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2.2. pack K
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_k) {
            auto value = ntt::inner_product(lhs, rhs);
            output = AccC ? output + value : value;
        }
        // 2.3. pack N
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_n) {
            output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
        }
        // 2.4. pack M & N
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mn ||
                           pack_kind == ukernels::mamtul_pack_kind::pack_kn) {
            auto value = ntt::outer_product(lhs, rhs);
            output = AccC ? output + value : value;
        }
        // 3.1. pack MK & K
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mk) {
            for (size_t m = 0; m < lhs.shape()[0]; m++) {
                auto value = ntt::inner_product(lhs(m), rhs);
                output(m) = AccC ? output(m) + value : value;
            }
        }
        // 3.2. pack MK & KN
        else if constexpr (pack_kind == ukernels::mamtul_pack_kind::pack_mkn) {
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
