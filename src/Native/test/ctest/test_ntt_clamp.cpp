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

TEST(ClampTestFloat, NoPack) {
    using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
    typeNoPack ti, to;
    std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
    std::fill(to.elements().begin(), to.elements().end(), 0.f);

    ntt::clamp(ti, to, -30.f, 30.f);

    bool result = (to(0, 0) == -30.f) && (to(0, 1) == -30.f) &&
                  (to(0, 2) == -30.f) && (to(0, 3) == -29.f) &&
                  (to(0, 4) == -28.f) && (to(7, 3) == 27.f) &&
                  (to(7, 4) == 28.f) && (to(7, 5) == 29.f) &&
                  (to(7, 6) == 30.f) && (to(7, 7) == 30.f);

    EXPECT_TRUE(result);
}

TEST(ClampTestFloat, PackM) {
    using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
    typeNoPack ti, to;
    std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
    std::fill(to.elements().begin(), to.elements().end(), 0.f);

    using typePack = ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<8, 1>>;

    typePack pi, po;
    ntt::pack<1>(ti, pi);
    ntt::pack<1>(to, po);

    ntt::clamp(pi, po, -30.f, 30.f);

    bool result = (po(0, 0)(0) == -30.f) && (po(0, 0)(1) == -30.f) &&
                  (po(0, 0)(2) == -30.f) && (po(0, 0)(3) == -29.f) &&
                  (po(0, 0)(4) == -28.f) && (po(7, 0)(3) == 27.f) &&
                  (po(7, 0)(4) == 28.f) && (po(7, 0)(5) == 29.f) &&
                  (po(7, 0)(6) == 30.f) && (po(7, 0)(7) == 30.f);

    EXPECT_TRUE(result);
}

TEST(ClampTestFloat, PackN) {
    using typeNoPack = ntt::tensor<float, ntt::fixed_shape<8, 8>>;
    typeNoPack ti, to;
    std::iota(ti.elements().begin(), ti.elements().end(), -32.f);
    std::fill(to.elements().begin(), to.elements().end(), 0.f);

    using typePack = ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 8>>;

    typePack pi, po;
    ntt::pack<0>(ti, pi);
    ntt::pack<0>(to, po);

    ntt::clamp(pi, po, -30.f, 30.f);

    bool result = (po(0, 0)(0) == -30.f) && (po(0, 0)(1) == -24.f) &&
                  (po(0, 0)(2) == -16.f) && (po(0, 0)(3) == -8.f) &&
                  (po(0, 0)(4) == -0.f) && (po(0, 7)(3) == -1.f) &&
                  (po(0, 7)(4) == 7.f) && (po(0, 7)(5) == 15.f) &&
                  (po(0, 7)(6) == 23.f) && (po(0, 7)(7) == 30.f);

    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
