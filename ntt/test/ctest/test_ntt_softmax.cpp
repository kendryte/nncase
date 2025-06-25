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

TEST(FixedShapePackedSoftmax, NoPack0) {
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    packed_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, NoPack1) {
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    packed_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16 / P, 16>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16 / P, 16>);
    packed_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16, 16 / P>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16, 16 / P>);
    packed_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsNotPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16 / P, 16>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16 / P, 16>);
    packed_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsNotPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16, 16 / P>);

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<3, 16, 16 / P>);
    packed_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16, 16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, NoPack0) {

    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    packed_softmax(buffer_1, ntt_output, 1_dim, ntt::fixed_shape_v<>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, NoPack1) {

    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);

    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    packed_softmax(buffer_1, ntt_output, 2_dim, ntt::fixed_shape_v<>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16 / P, 16));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16 / P, 16));
    packed_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<1>);
    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16, 16 / P));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16, 16 / P));
    packed_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<2>);
    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsNotPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16 / P, 16));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<1>);
    auto buffer_3 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16 / P, 16));
    packed_softmax(buffer_2, buffer_3, 2_dim, ntt::fixed_shape_v<1>);
    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsNotPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    auto buffer_1 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::init_tensor(buffer_1, -10.f, 10.f);
    auto buffer_2 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16, 16 / P));

    pack(buffer_1, buffer_2, ntt::fixed_shape_v<2>);
    auto buffer_3 =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(3, 16, 16 / P));
    packed_softmax(buffer_2, buffer_3, 1_dim, ntt::fixed_shape_v<2>);
    auto ntt_output = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    unpack(buffer_3, ntt_output, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(3, 16, 16));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
