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
#include "../primitive_ops.h"
#include "../vector.h"

namespace nncase::ntt {
namespace ukernels {
template <bool AccumulateC, dim_t M0Tile, Scalar TAElem, Vector TBPack,
          Vector TCPack, bool Arch>
struct u_packed_matmul {
    using TBElem = replace_lanes_t<TBPack, TBPack::shape()[1_dim]>;
    using TCElem = replace_lanes_t<TCPack, TCPack::shape()[1_dim]>;
    static constexpr auto N0Tile = TCPack::shape()[0_dim];

    template <Dimension TLda, Dimension TLdc, Dimension TK>
    constexpr void operator()(const TAElem *NTT_RESTRICT a,
                              const TBPack *NTT_RESTRICT b,
                              TCPack *NTT_RESTRICT c, const TLda &lda,
                              const TLdc &ldc, const TK &K) noexcept {
        TCElem c0_tmp[M0Tile][N0Tile];
        ntt::apply(fixed_shape_v<M0Tile, N0Tile>, [&](auto index) {
            c0_tmp[index[0_dim]][index[1_dim]] =
                AccumulateC ? c[index[0_dim] * ldc](index[1_dim]) : TCElem{};
        });

        for (size_t k1 = 0; k1 < K; k1++) {
            TAElem a0_tmp[M0Tile];
            TBElem b0_tmp[N0Tile];

            ntt::apply(fixed_shape_v<M0Tile>, [&](auto index) {
                a0_tmp[index[0_dim]] = a[index[0_dim] * lda + k1];
            });

            ntt::apply(fixed_shape_v<N0Tile>, [&](auto index) {
                b0_tmp[index[0_dim]] = b[k1](index[0_dim]);
            });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < M0Tile; m++) {
                    c0_tmp[m][n] =
                        ntt::mul_add(a0_tmp[m], b0_tmp[n], c0_tmp[m][n]);
                }
            }
        }

        ntt::apply(fixed_shape_v<M0Tile, N0Tile>, [&](auto index) {
            ntt::store(c[index[0_dim] * ldc](index[1_dim]),
                       c0_tmp[index[0_dim]][index[1_dim]]);
        });
    }
};

} // namespace ukernels

template <bool AccumulateC, dim_t M0Tile, Scalar TAElem, Vector TBPack,
          Vector TCPack, Dimension TLda, Dimension TLdc, Dimension TK>
constexpr void u_packed_matmul(const TAElem *a, const TBPack *b, TCPack *c,
                               const TLda &lda, const TLdc &ldc,
                               const TK &K) noexcept {
    ukernels::u_packed_matmul<AccumulateC, M0Tile, TAElem, TBPack, TCPack, true>
        impl;
    impl(a, b, c, lda, ldc, K);
}
} // namespace nncase::ntt
