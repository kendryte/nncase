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
#include "matmul.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <bool AccumulateC, class TLhs, class TRhs, class TOut>
class packed_matmul_impl;

/**
 * @brief 1D-vectorized matmul with packed B.
 * @remarks Loop orders: (m, n, k)
 */
template <bool AccumulateC, ValidMatmulTensor TLhs, ValidMatmulTensor TRhs,
          ValidMatmulTensor TOut>
class packed_matmul_impl<AccumulateC, TLhs, TRhs, TOut> {
    using TOutElem = typename TOut::element_type;

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
        constexpr auto m0_tile = 2;

        for (size_t n1 = 0; n1 < N; n1++) {
            for (size_t m1 = 0; m1 < M; m1 += m0_tile) {
                matmul_2d_l0<m0_tile>(a, b, c, K, m1, n1);
            }
        }

        if (M % m0_tile) {
            for (size_t n1 = 0; n1 < N; n1++) {
                for (size_t m1 = M / m0_tile * m0_tile; m1 < M; m1 += m0_tile) {
                    matmul_2d_l0<m0_tile>(a, b, c, K, m1, n1);
                }
            }
        }
    }

    template <dim_t M0Tile, class TA, class TB, class TC, Dimension TK>
    void matmul_2d_l0(const TA &a, const TB &b, TC &c, const TK &K, dim_t m1,
                      dim_t n1) {
        auto c0 = c.view(make_shape(m1, n1), fixed_shape_v<M0Tile, 1_dim>);
        auto a1 =
            a.view(make_shape(m1, 0_dim), make_shape(fixed_dim_v<M0Tile>, K));
        auto b1 = b.view(make_shape(n1, 0_dim), make_shape(1_dim, K));
        ntt::u_packed_matmul<AccumulateC, M0Tile>(a1, b1, c0, K);
    }
};
} // namespace detail

/**
 * @brief packed matmul
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
void packed_matmul(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    detail::packed_matmul_impl<AccumulateC, TLhs, TRhs, std::decay_t<TOut>>
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
