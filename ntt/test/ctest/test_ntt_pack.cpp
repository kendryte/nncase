/* Copyright 2019-2024 Canaan Inc.
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
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(GatherTestFloat, pack1d_dim0) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = 64;
    constexpr size_t N = 64;
    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>
        pa;
    ntt::pack<0>(ta, pa);
    ntt::unpack<0>(pa, tc);
    EXPECT_TRUE(NttTest::compare_tensor(ta, tc));
}

TEST(GatherTestFloat, pack1d_dim1) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = 64;
    constexpr size_t N = 64;
    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>
        pa;
    ntt::pack<1>(ta, pa);
    ntt::unpack<1>(pa, tc);
    EXPECT_TRUE(NttTest::compare_tensor(ta, tc));
}

TEST(GatherTestFloat, pack2d) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = 64;
    constexpr size_t N = 64;
    ntt::tensor<float, ntt::fixed_shape<M, N>> ta;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<M / P, N / P>>
            pa;
    ntt::pack<0, 1>(ta, pa);
    ntt::unpack<0, 1>(pa, tc);
    EXPECT_TRUE(NttTest::compare_tensor(ta, tc));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}