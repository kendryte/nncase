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

TEST(FixedShapeLayerNorm, NoPack0) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no pack
    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> ntt_output;
    packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    const float array_golden[] = {
        0.000000f,  -0.570438f, -0.924264f, -1.061478f, -0.982080f, -0.686069f,
        -0.173447f, 0.555789f,  1.501636f,  2.664094f,  4.043166f,  5.638849f,
        7.451145f,  9.480053f,  11.725573f, 14.187704f, 16.866449f, 19.761805f,
        22.873774f, 26.202354f, 29.747547f, 33.509354f, 37.487770f, 41.682800f,
        46.094440f, 50.722694f, 55.567558f, 60.629036f, 65.907127f, 71.401825f,
        77.113144f, 83.041069f};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, NoPack1) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    // no pack
    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> ntt_output;
    packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    ntt::apply(ntt_output.shape(), [&](auto index) {
        std::cout << std::setprecision(6) << std::fixed << "ntt_output("
                  << index[0] << ", " << index[1] << ", " << index[2]
                  << ") = " << ntt_output(index) << std::endl;
    });

    const float array_golden[] = {
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 4>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, NoPack2) {
    ntt::tensor<float, ntt::fixed_shape<1, 13, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<13, 2>> buffer_4;
    ntt::tensor<float, ntt::fixed_shape<13, 2>> buffer_7;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_4.elements().begin(), buffer_4.elements().end(), 0.f);
    std::iota(buffer_7.elements().begin(), buffer_7.elements().end(), 0.f);

    // no pack with pad
    ntt::tensor<float, ntt::fixed_shape<1, 13, 2>> ntt_output;
    packed_layer_norm<1>(buffer_1, buffer_4, buffer_7, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    const float array_golden[] = {
        0.000000,  -0.533333, -0.800000, -0.800000, -0.533334, -0.000000,
        0.800000,  1.866666,  3.200000,  4.800000,  6.666667,  8.799999,
        11.200000, 13.866667, 16.799999, 20.000000, 23.466667, 27.200001,
        31.200001, 35.466667, 40.000000, 44.800003, 49.866669, 55.200001,
        60.800003, 66.666672};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 13, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, Pack0) {
    // packed axis == layer norm axis

    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_3;
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> buffer_4;
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> buffer_5;
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_6;
    pack<1>(buffer_0, buffer_3);
    pack<0>(buffer_1, buffer_4);
    pack<0>(buffer_2, buffer_5);
    packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                         ntt::vector<float, 8>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> ntt_output;
    unpack<1>(buffer_6, ntt_output);

    const float array_golden[] = {
        0.000000f,  -0.570438f, -0.924264f, -1.061478f, -0.982080f, -0.686069f,
        -0.173447f, 0.555789f,  1.501636f,  2.664094f,  4.043166f,  5.638849f,
        7.451145f,  9.480053f,  11.725573f, 14.187704f, 16.866449f, 19.761805f,
        22.873774f, 26.202354f, 29.747547f, 33.509354f, 37.487770f, 41.682800f,
        46.094440f, 50.722694f, 55.567558f, 60.629036f, 65.907127f, 71.401825f,
        77.113144f, 83.041069f};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, Pack1) {

    // packed axis == layer norm axis

    ntt::tensor<float, ntt::fixed_shape<1, 2, 16>> input;
    ntt::tensor<float, ntt::fixed_shape<16>> scale;
    ntt::tensor<float, ntt::fixed_shape<16>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 2, 4>> input_packed;
    ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<4>> scale_packed;
    ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<4>> bias_packed;
    ntt::pack<2>(input, input_packed);
    ntt::pack<0>(scale, scale_packed);
    ntt::pack<0>(bias, bias_packed);
    ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 2, 4>> output_packed;
    packed_layer_norm<2>(input_packed, scale_packed, bias_packed, output_packed,
                         ntt::vector<float, 4>::from_scalar(1E-06), true,
                         ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 2, 16>> ntt_output;
    unpack<2>(output_packed, ntt_output);

    const float array_golden[] = {
        15.000000, 12.589952, 10.613765, 9.071439,  7.962974,  7.288369,
        7.047626,  7.240744,  7.867722,  8.928561,  10.423262, 12.351823,
        14.714245, 17.510529, 20.740673, 24.404676, 15.000000, 12.589952,
        10.613765, 9.071439,  7.962974,  7.288369,  7.047626,  7.240744,
        7.867722,  8.928561,  10.423262, 12.351823, 14.714245, 17.510529,
        20.740673, 24.404676};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 2, 16>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, Pack2) {
    // packed axis < layer norm axis

    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 4>> buffer_3;
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 4>> buffer_4;
    pack<1>(buffer_0, buffer_3);
    packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                         ntt::vector<float, 8>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> ntt_output;
    unpack<1>(buffer_4, ntt_output);

    const float array_golden[] = {
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562, -0.341640, 1.105573,
        4.341640,  9.366562, -0.341640, 1.105573, 4.341640,  9.366562,
        -0.341640, 1.105573, 4.341640,  9.366562};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 4>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeLayerNorm, Pack3) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> input;
    ntt::tensor<float, ntt::fixed_shape<8>> scale;
    ntt::tensor<float, ntt::fixed_shape<8>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 8>> packed_input;
    ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 8>> packed_output;
    pack<1>(input, packed_input);
    packed_layer_norm<2>(packed_input, scale, bias, packed_output,
                         ntt::vector<float, 8>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> ntt_output;
    unpack<1>(packed_output, ntt_output);

    const float array_golden[] = {
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677, 0.000000,  -0.091089, 0.690693,  2.345346,
        4.872871,  8.273268,  12.546536, 17.692677, 0.000000,  -0.091089,
        0.690693,  2.345346,  4.872871,  8.273268,  12.546536, 17.692677,
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677, 0.000000,  -0.091089, 0.690693,  2.345346,
        4.872871,  8.273268,  12.546536, 17.692677, 0.000000,  -0.091089,
        0.690693,  2.345346,  4.872871,  8.273268,  12.546536, 17.692677,
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677, 0.000000,  -0.091089, 0.690693,  2.345346,
        4.872871,  8.273268,  12.546536, 17.692677, 0.000000,  -0.091089,
        0.690693,  2.345346,  4.872871,  8.273268,  12.546536, 17.692677,
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677, 0.000000,  -0.091089, 0.690693,  2.345346,
        4.872871,  8.273268,  12.546536, 17.692677, 0.000000,  -0.091089,
        0.690693,  2.345346,  4.872871,  8.273268,  12.546536, 17.692677,
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677, 0.000000,  -0.091089, 0.690693,  2.345346,
        4.872871,  8.273268,  12.546536, 17.692677, 0.000000,  -0.091089,
        0.690693,  2.345346,  4.872871,  8.273268,  12.546536, 17.692677,
        0.000000,  -0.091089, 0.690693,  2.345346,  4.872871,  8.273268,
        12.546536, 17.692677};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 8>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
