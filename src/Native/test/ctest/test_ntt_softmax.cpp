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

TEST(SoftMax, NoPackDim0) {

    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 0);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    packed_softmax<0>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    EXPECT_TRUE(NttTest::compare_tensor(buffer_2, buffer_2_ort));
}

TEST(SoftMax, NoPackDim1) {

    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 1);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    packed_softmax<1>(buffer_1, buffer_2, ntt::fixed_shape<>{});
    EXPECT_TRUE(NttTest::compare_tensor(buffer_2, buffer_2_ort));
}

TEST(SoftMax, Pack0Dim0) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<0>(buffer_1, buffer_1_p);
    packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    unpack<0>(buffer_2_p, buffer_2_up);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 0);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    EXPECT_TRUE(NttTest::compare_tensor(buffer_2_up, buffer_2_ort));
}

TEST(SoftMax, Pack0Dim1) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<0>(buffer_1, buffer_1_p);
    packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<0>{});
    unpack<0>(buffer_2_p, buffer_2_up);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 1);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    EXPECT_TRUE(NttTest::compare_tensor(buffer_2_up, buffer_2_ort));
}

TEST(SoftMax, Pack1Dim0) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<1>(buffer_1, buffer_1_p);
    packed_softmax<0>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    unpack<1>(buffer_2_p, buffer_2_up);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 0);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    EXPECT_TRUE(NttTest::compare_tensor(buffer_2_up, buffer_2_ort));
}

TEST(SoftMax, Pack1Dim1) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_ort;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_1_p;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>> buffer_2_p;
    ntt::tensor<float, ntt::fixed_shape<32, 32>> buffer_2_up;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    pack<1>(buffer_1, buffer_1_p);
    packed_softmax<1>(buffer_1_p, buffer_2_p, ntt::fixed_shape<1>{});
    unpack<1>(buffer_2_p, buffer_2_up);

    auto ort_buffer_1 = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_buffer_1, 1);
    NttTest::ort2ntt(ort_output, buffer_2_ort);

    EXPECT_TRUE(NttTest::compare_tensor(buffer_2_up, buffer_2_ort));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
