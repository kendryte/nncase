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

TEST(PackedSoftmax, NoPack0) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output;
    packed_softmax<1>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(PackedSoftmax, NoPack1) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output;
    packed_softmax<2>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(PackedSoftmax, AxisIsPackedAxis0) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 16>> buffer_2;

    pack<1>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 16>> buffer_3;
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output;
    unpack<1>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(PackedSoftmax, AxisIsPackedAxis1) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 16, 2>> buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 16, 2>> buffer_3;
    packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output;
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    ntt::tensor<float, ntt::fixed_shape<1, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
