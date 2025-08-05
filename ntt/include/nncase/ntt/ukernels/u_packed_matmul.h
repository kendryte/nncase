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
#include "../vector.h"
#include "u_mul_add.h"
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {
template <bool AccumulateC, dim_t M0Tile, class TLhsElem, class TRhsElem,
          class TOutElem, bool Arch>
struct u_packed_matmul {
    static constexpr auto N0Tile = TRhsElem::shape()[0_dim];

    template <class TA, class TB, class TC>
    constexpr void operator()(const TA &a, const TB &b, TC &c0,
                              dim_t K) noexcept {
        TOutElem c0_tmp[M0Tile];
        ntt::apply(c0.shape(), [&](auto index) {
            c0_tmp[index[0_dim]] = AccumulateC ? c0(index) : TOutElem{};
        });

        for (size_t k1 = 0; k1 < K; k1++) {
            auto a0 = a.view(make_shape(0, k1), fixed_shape_v<M0Tile, 1>);
            auto b0 = b(0_dim, k1);
            TLhsElem a0_tmp[M0Tile];

            ntt::apply(fixed_shape_v<M0Tile>, [&](auto index) {
                a0_tmp[index[0_dim]] = a0(index[0_dim], 0_dim);
            });

            for (size_t n = 0; n < N0Tile; n++) {
                for (size_t m = 0; m < M0Tile; m++) {
                    u_mul_add<ukernels::matmul_vectorize_kind::vectorize_n,
                              true>(a0_tmp[m], b0(n), c0_tmp[m](n));
                }
            }
        }

        ntt::apply(c0.shape(), [&](auto index) {
            ntt::store(c0(index), c0_tmp[index[0_dim]]);
        });
    }
};

} // namespace ukernels

template <bool AccumulateC, dim_t M0Tile, class TA, class TB, class TC,
          Dimension TK>
constexpr void u_packed_matmul(const TA &a, const TB &b, TC &c,
                               const TK &K) noexcept {
    using TLhsElem = std::decay_t<typename TA::element_type>;
    using TRhsElem = std::decay_t<typename TB::element_type>;
    using TOutElem = std::decay_t<typename TC::element_type>;
    ukernels::u_packed_matmul<AccumulateC, M0Tile, TLhsElem, TRhsElem, TOutElem,
                              true>
        impl;
    impl(a, b, c, K);
}
} // namespace nncase::ntt
