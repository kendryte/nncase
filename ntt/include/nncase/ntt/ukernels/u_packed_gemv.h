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

namespace nncase::ntt {
namespace ukernels {
template <bool AccumulateC, Scalar TAElem, Vector TBPack, Vector TCPack,
          bool Arch>
struct u_packed_gemv {
    static constexpr auto N0Tile = TCPack::shape()[0_dim];

    template <Dimension TLdb, Dimension TK, Dimension TN>
    constexpr void operator()(const TAElem *NTT_RESTRICT a,
                              const TBPack *NTT_RESTRICT b,
                              TCPack *NTT_RESTRICT c, const TLdb &ldb,
                              const TK &K, const TN &N) noexcept {
        if constexpr (!AccumulateC) {
            for (size_t n1 = 0; n1 < N; n1++) {
                c[n1] = {};
            }
        }

        for (size_t n1 = 0; n1 < N; n1++) {
            const auto b1 = b + n1 * ldb;
            TCPack c0 = c[n1];

            for (size_t k1 = 0; k1 < K; k1++) {
                const TAElem a0 = a[k1];
                const auto b0 = b1[k1];
                ntt::apply(fixed_shape_v<N0Tile>, [&](auto index) {
                    c0(index[0_dim]) =
                        ntt::mul_add(a0, b0(index[0_dim]), c0(index[0_dim]));
                });
            }

            ntt::apply(fixed_shape_v<N0Tile>, [&](auto index) {
                c[n1](index[0_dim]) = c0(index[0_dim]);
            });
        }
    }
};

} // namespace ukernels

template <bool AccumulateC, Scalar TAElem, Vector TBPack, Vector TCPack,
          Dimension TLdb, Dimension TK, Dimension TN>
constexpr void u_packed_gemv(const TAElem *a, const TBPack *b, TCPack *c,
                             const TLdb &ldb, const TK &K,
                             const TN &N) noexcept {
    ukernels::u_packed_gemv<AccumulateC, TAElem, TBPack, TCPack, true> impl;
    impl(a, b, c, ldb, K, N);
}
} // namespace nncase::ntt
