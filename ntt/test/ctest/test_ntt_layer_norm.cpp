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
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 2>> buffer_3;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16 / P, 2>> buffer_4;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16 / P, 2>> buffer_5;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 2>> buffer_6;
    pack<1>(buffer_0, buffer_3);
    pack<0>(buffer_1, buffer_4);
    pack<0>(buffer_2, buffer_5);
    packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
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
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    ntt::tensor<float, ntt::fixed_shape<1, 2, 16>> input;
    ntt::tensor<float, ntt::fixed_shape<16>> scale;
    ntt::tensor<float, ntt::fixed_shape<16>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 2, 16 / P>>
        input_packed;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16 / P>> scale_packed;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16 / P>> bias_packed;
    ntt::pack<2>(input, input_packed);
    ntt::pack<0>(scale, scale_packed);
    ntt::pack<0>(bias, bias_packed);
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 2, 16 / P>>
        output_packed;
    packed_layer_norm<2>(input_packed, scale_packed, bias_packed, output_packed,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
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
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 4>> buffer_3;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 4>> buffer_4;
    pack<1>(buffer_0, buffer_3);
    packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
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
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> input;
    ntt::tensor<float, ntt::fixed_shape<8>> scale;
    ntt::tensor<float, ntt::fixed_shape<8>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 8>>
        packed_input;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16 / P, 8>>
        packed_output;
    pack<1>(input, packed_input);
    packed_layer_norm<2>(packed_input, scale, bias, packed_output,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
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

TEST(FixedShapeLayerNorm, Pack4) {
    // packed axis > layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> input;
    ntt::tensor<float, ntt::fixed_shape<16, 8>> scale;
    ntt::tensor<float, ntt::fixed_shape<16, 8>> bias;
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16, 8 / P>>
        packed_input;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16, 8 / P>>
        packed_scale;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<16, 8 / P>> packed_bias;
    ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<1, 16, 8 / P>>
        packed_output;
    pack<2>(input, packed_input);
    pack<1>(scale, packed_scale);
    pack<1>(bias, packed_bias);
    packed_layer_norm<1>(packed_input, packed_scale, packed_bias, packed_output,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> ntt_output;
    unpack<2>(packed_output, ntt_output);

    const float array_golden[] = {
        0.000000,   -0.691507,  -1.328887,  -1.912138,  -2.441260,  -2.916255,
        -3.337121,  -3.703859,  -4.016469,  -4.274950,  -4.479303,  -4.629529,
        -4.725626,  -4.767593,  -4.755434,  -4.689146,  -4.568729,  -4.394186,
        -4.165514,  -3.882713,  -3.545782,  -3.154726,  -2.709541,  -2.210226,
        -1.656784,  -1.049215,  -0.387516,  0.328310,   1.098267,   1.922350,
        2.800560,   3.732901,   4.719368,   5.759966,   6.854689,   8.003542,
        9.206524,   10.463631,  11.774870,  13.140234,  14.559729,  16.033350,
        17.561102,  19.142979,  20.778986,  22.469122,  24.213385,  26.011776,
        27.864296,  29.770945,  31.731720,  33.746624,  35.815659,  37.938820,
        40.116108,  42.347527,  44.633072,  46.972744,  49.366547,  51.814476,
        54.316536,  56.872723,  59.483036,  62.147480,  64.866051,  67.638748,
        70.465576,  73.346535,  76.281616,  79.270836,  82.314178,  85.411644,
        88.563240,  91.768967,  95.028824,  98.342804,  101.710915, 105.133148,
        108.609520, 112.140015, 115.724640, 119.363388, 123.056267, 126.803276,
        130.604416, 134.459671, 138.369080, 142.332581, 146.350250, 150.422012,
        154.547913, 158.727951, 162.962128, 167.250412, 171.592834, 175.989380,
        180.440048, 184.944855, 189.503784, 194.116852, 198.784027, 203.505356,
        208.280792, 213.110367, 217.994064, 222.931900, 227.923859, 232.969940,
        238.070160, 243.224503, 248.432968, 253.695557, 259.012299, 264.383148,
        269.808105, 275.287231, 280.820496, 286.407837, 292.049347, 297.744995,
        303.494720, 309.298584, 315.156616, 321.068756, 327.035034, 333.055420,
        339.129944, 345.258606};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 8>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, NoPack0) {

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 2);
    auto shape2 = ntt::make_ranked_shape(16, 2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no pack
    tensor_type1 ntt_output(shape1);
    packed_layer_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    tensor_type1 ntt_golden(shape1);
    float array_golden[] = {
        0.000000f,  -0.570438f, -0.924264f, -1.061478f, -0.982080f, -0.686069f,
        -0.173447f, 0.555789f,  1.501636f,  2.664094f,  4.043166f,  5.638849f,
        7.451145f,  9.480053f,  11.725573f, 14.187704f, 16.866449f, 19.761805f,
        22.873774f, 26.202354f, 29.747547f, 33.509354f, 37.487770f, 41.682800f,
        46.094440f, 50.722694f, 55.567558f, 60.629036f, 65.907127f, 71.401825f,
        77.113144f, 83.041069f};
    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, NoPack1) {
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 4);
    auto shape2 = ntt::make_ranked_shape(4);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    // no pack
    tensor_type1 ntt_output(shape1);
    packed_layer_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    ntt::apply(ntt_output.shape(), [&](auto index) {
        std::cout << std::setprecision(6) << std::fixed << "ntt_output("
                  << index[0] << ", " << index[1] << ", " << index[2]
                  << ") = " << ntt_output(index) << std::endl;
    });

    tensor_type1 ntt_golden(shape1);
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

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, NoPack2) {

    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(1, 13, 2);
    auto shape2 = ntt::make_ranked_shape(13, 2);

    tensor_type1 buffer_1(shape1);
    tensor_type2 buffer_4(shape2);
    tensor_type2 buffer_7(shape2);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_4.elements().begin(), buffer_4.elements().end(), 0.f);
    std::iota(buffer_7.elements().begin(), buffer_7.elements().end(), 0.f);

    // no pack with pad
    tensor_type1 ntt_output(shape1);
    packed_layer_norm<1>(buffer_1, buffer_4, buffer_7, ntt_output, 1e-06, true,
                         ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,  -0.533333, -0.800000, -0.800000, -0.533334, -0.000000,
        0.800000,  1.866666,  3.200000,  4.800000,  6.666667,  8.799999,
        11.200000, 13.866667, 16.799999, 20.000000, 23.466667, 27.200001,
        31.200001, 35.466667, 40.000000, 44.800003, 49.866669, 55.200001,
        60.800003, 66.666672};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, Pack0) {
    // packed axis == layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 2);
    auto shape2 = ntt::make_ranked_shape(16, 2);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<2>>;
    auto shape3 = ntt::make_ranked_shape(1, 2, 2);
    auto shape4 = ntt::make_ranked_shape(2, 2);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    tensor_type3 buffer_3(shape3);
    tensor_type4 buffer_4(shape4);
    tensor_type4 buffer_5(shape4);
    tensor_type3 buffer_6(shape3);
    pack<1>(buffer_0, buffer_3);
    pack<0>(buffer_1, buffer_4);
    pack<0>(buffer_2, buffer_5);
    packed_layer_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_6, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000f,  -0.570438f, -0.924264f, -1.061478f, -0.982080f, -0.686069f,
        -0.173447f, 0.555789f,  1.501636f,  2.664094f,  4.043166f,  5.638849f,
        7.451145f,  9.480053f,  11.725573f, 14.187704f, 16.866449f, 19.761805f,
        22.873774f, 26.202354f, 29.747547f, 33.509354f, 37.487770f, 41.682800f,
        46.094440f, 50.722694f, 55.567558f, 60.629036f, 65.907127f, 71.401825f,
        77.113144f, 83.041069f};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, Pack1) {

    // packed axis == layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(1, 2, 16);
    auto shape2 = ntt::make_ranked_shape(16);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<1>>;
    auto shape3 = ntt::make_ranked_shape(1, 2, 16 / P);
    auto shape4 = ntt::make_ranked_shape(16 / P);

    tensor_type1 input(shape1);
    tensor_type2 scale(shape2);
    tensor_type2 bias(shape2);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().rbegin(), bias.elements().rend(), 0.f);

    tensor_type3 input_packed(shape3);
    tensor_type4 scale_packed(shape4);
    tensor_type4 bias_packed(shape4);
    ntt::pack<2>(input, input_packed);
    ntt::pack<0>(scale, scale_packed);
    ntt::pack<0>(bias, bias_packed);
    tensor_type3 output_packed(shape3);
    packed_layer_norm<2>(input_packed, scale_packed, bias_packed, output_packed,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<2>(output_packed, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        15.000000, 12.589952, 10.613765, 9.071439,  7.962974,  7.288369,
        7.047626,  7.240744,  7.867722,  8.928561,  10.423262, 12.351823,
        14.714245, 17.510529, 20.740673, 24.404676, 15.000000, 12.589952,
        10.613765, 9.071439,  7.962974,  7.288369,  7.047626,  7.240744,
        7.867722,  8.928561,  10.423262, 12.351823, 14.714245, 17.510529,
        20.740673, 24.404676};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, Pack2) {
    // packed axis < layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 4);
    auto shape2 = ntt::make_ranked_shape(4);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<1>>;
    auto shape3 = ntt::make_ranked_shape(1, 16 / P, 4);

    tensor_type1 buffer_0(shape1);
    tensor_type2 buffer_1(shape2);
    tensor_type2 buffer_2(shape2);
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    tensor_type3 buffer_3(shape3);
    tensor_type3 buffer_4(shape3);
    pack<1>(buffer_0, buffer_3);
    packed_layer_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_4, ntt_output);

    tensor_type1 ntt_golden(shape1);
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

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, Pack3) {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<1>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 8);
    auto shape2 = ntt::make_ranked_shape(8);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<1>>;
    auto shape3 = ntt::make_ranked_shape(1, 16 / P, 8);

    tensor_type1 input(shape1);
    tensor_type2 scale(shape2);
    tensor_type2 bias(shape2);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    tensor_type3 packed_input(shape3);
    tensor_type3 packed_output(shape3);
    pack<1>(input, packed_input);
    packed_layer_norm<2>(packed_input, scale, bias, packed_output,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(packed_output, ntt_output);

    tensor_type1 ntt_golden(shape1);
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

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeLayerNorm, Pack4) {
    // packed axis > layer norm axis
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<3>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<2>>;
    auto shape1 = ntt::make_ranked_shape(1, 16, 8);
    auto shape2 = ntt::make_ranked_shape(16, 8);

    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<3>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<2>>;
    auto shape3 = ntt::make_ranked_shape(1, 16, 8 / P);
    auto shape4 = ntt::make_ranked_shape(16, 8 / P);

    tensor_type1 input(shape1);
    tensor_type2 scale(shape2);
    tensor_type2 bias(shape2);
    std::iota(input.elements().begin(), input.elements().end(), 0.f);
    std::iota(scale.elements().begin(), scale.elements().end(), 0.f);
    std::iota(bias.elements().begin(), bias.elements().end(), 0.f);

    // packed axis < layer norm axis
    tensor_type3 packed_input(shape3);
    tensor_type4 packed_scale(shape4);
    tensor_type4 packed_bias(shape4);
    tensor_type3 packed_output(shape3);
    pack<2>(input, packed_input);
    pack<1>(scale, packed_scale);
    pack<1>(bias, packed_bias);
    packed_layer_norm<1>(packed_input, packed_scale, packed_bias, packed_output,
                         ntt::vector<float, P>::from_scalar(1E-06), true,
                         ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<2>(packed_output, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,   -0.691507,  -1.328887,  -1.912138,  -2.441260,  -2.916255,
        -3.337121,  -3.703859,  -4.016469,  -4.274950,  -4.479303,  -4.629529,
        -4.725626,  -4.767593,  -4.755434,  -4.689146,  -4.568729,  -4.394186,
        -4.165514,  -3.882713,  -3.545782,  -3.154726,  -2.709541,  -2.210226,
        -1.656784,  -1.049215,  -0.387516,  0.328310,   1.098267,   1.922350,
        2.800560,   3.732901,   4.719368,   5.759966,   6.854689,   8.003542,
        9.206524,   10.463631,  11.774870,  13.140234,  14.559729,  16.033350,
        17.561102,  19.142979,  20.778986,  22.469122,  24.213385,  26.011776,
        27.864296,  29.770945,  31.731720,  33.746624,  35.815659,  37.938820,
        40.116108,  42.347527,  44.633072,  46.972744,  49.366547,  51.814476,
        54.316536,  56.872723,  59.483036,  62.147480,  64.866051,  67.638748,
        70.465576,  73.346535,  76.281616,  79.270836,  82.314178,  85.411644,
        88.563240,  91.768967,  95.028824,  98.342804,  101.710915, 105.133148,
        108.609520, 112.140015, 115.724640, 119.363388, 123.056267, 126.803276,
        130.604416, 134.459671, 138.369080, 142.332581, 146.350250, 150.422012,
        154.547913, 158.727951, 162.962128, 167.250412, 171.592834, 175.989380,
        180.440048, 184.944855, 189.503784, 194.116852, 198.784027, 203.505356,
        208.280792, 213.110367, 217.994064, 222.931900, 227.923859, 232.969940,
        238.070160, 243.224503, 248.432968, 253.695557, 259.012299, 264.383148,
        269.808105, 275.287231, 280.820496, 286.407837, 292.049347, 297.744995,
        303.494720, 309.298584, 315.156616, 321.068756, 327.035034, 333.055420,
        339.129944, 345.258606};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
