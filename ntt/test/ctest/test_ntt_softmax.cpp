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
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    packed_softmax<1>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, NoPack1) {
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    packed_softmax<2>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16 / P, 16>>
        buffer_2;

    pack<1>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16 / P, 16>>
        buffer_3;
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    unpack<1>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_3;
    packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsNotPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16 / P, 16>>
        buffer_2;

    pack<1>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16 / P, 16>>
        buffer_3;
    packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<3>{});
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    unpack<1>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsNotPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_3;
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(FixedShapePackedSoftmax, AxisIsNotPackedAxis1Debug) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> buffer_1;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_2;

    pack<2>(buffer_1, buffer_2);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<3, 16, 16 / P>>
        buffer_3;
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output;
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    ntt::tensor<float, ntt::fixed_shape<3, 16, 16>> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);

    std::cout << "ntt_output: " << std::endl;
    ntt::apply(ntt_output.shape(), [&](auto index) {
        std::cout << ntt_output(index) << " " << std::endl;
    });
    std::cout << "ntt_output2: " << std::endl;
    ntt::apply(ntt_output2.shape(), [&](auto index) {
        std::cout << ntt_output2(index) << " " << std::endl;
    });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, NoPack0) {

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    tensor_type1 ntt_output(shape1);
    packed_softmax<1>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, NoPack1) {

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);

    tensor_type1 ntt_output(shape1);
    packed_softmax<2>(buffer_1, ntt_output, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);
    auto shape2 = ntt::make_ranked_shape(3, 16 / P, 16);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    tensor_type2 buffer_2(shape2);

    pack<1>(buffer_1, buffer_2);
    tensor_type2 buffer_3(shape2);
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<1>{});
    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);
    auto shape2 = ntt::make_ranked_shape(3, 16, 16 / P);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    tensor_type2 buffer_2(shape2);

    pack<2>(buffer_1, buffer_2);
    tensor_type2 buffer_3(shape2);
    packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    tensor_type1 ntt_output(shape1);
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsNotPackedAxis0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);
    auto shape2 = ntt::make_ranked_shape(3, 16 / P, 16);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    tensor_type2 buffer_2(shape2);

    pack<1>(buffer_1, buffer_2);
    tensor_type2 buffer_3(shape2);
    packed_softmax<2>(buffer_2, buffer_3, ntt::fixed_shape<3>{});
    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 2);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(RankedShapePackedSoftmax, AxisIsNotPackedAxis1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    auto shape1 = ntt::make_ranked_shape(3, 16, 16);
    auto shape2 = ntt::make_ranked_shape(3, 16, 16 / P);

    tensor_type1 buffer_1(shape1);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    tensor_type2 buffer_2(shape2);

    pack<2>(buffer_1, buffer_2);
    tensor_type2 buffer_3(shape2);
    packed_softmax<1>(buffer_2, buffer_3, ntt::fixed_shape<2>{});
    tensor_type1 ntt_output(shape1);
    unpack<2>(buffer_3, ntt_output);

    // ort
    auto ort_input = NttTest::ntt2ort(buffer_1);
    auto ort_output = ortki_Softmax(ort_input, 1);

    // compare
    tensor_type1 ntt_output2(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
