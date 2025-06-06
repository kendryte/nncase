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

TEST(FixedShapeRMSNorm, NoPack0) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 0.f);

    // no pack
    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> ntt_output;
    packed_rms_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    const float array_golden[] = {
        0.000000,  1.055427,  2.221709,  3.498847,  4.886838,  6.385685,
        7.995387,  9.715942,  11.547354, 13.489619, 15.542740, 17.706715,
        19.981546, 22.367229, 24.863770, 27.471165, 30.189415, 33.018517,
        35.958477, 39.009293, 42.170959, 45.443481, 48.826862, 52.321095,
        55.926186, 59.642124, 63.468922, 67.406570, 71.455078, 75.614441,
        79.884659, 84.265732};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, NoPack1) {
    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> buffer_0;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<4>> buffer_2;
    std::iota(buffer_0.elements().begin(), buffer_0.elements().end(), 0.f);
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 1.f);
    std::iota(buffer_2.elements().begin(), buffer_2.elements().end(), 1.f);

    // no pack
    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> ntt_output;
    packed_rms_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    const float array_golden[] = {
        1.000000, 3.069045, 6.207134, 10.414268, 1.712697, 3.781742, 6.207135,
        8.988876, 1.836333, 3.881750, 6.136250,  8.599833, 1.885856, 3.919355,
        6.100497, 8.429281, 1.912426, 3.938904,  6.079436, 8.334021, 1.928977,
        3.950852, 6.065625, 8.273296, 1.940273,  3.958902, 6.055888, 8.231230,
        1.948472, 3.964691, 6.048659, 8.200375,  1.954692, 3.969053, 6.043082,
        8.176779, 1.959574, 3.972457, 6.038650,  8.158153, 1.963506, 3.975187,
        6.035044, 8.143075, 1.966741, 3.977425,  6.032052, 8.130621, 1.969450,
        3.979293, 6.029531, 8.120161, 1.971750,  3.980876, 6.027376, 8.111252,
        1.973729, 3.982234, 6.025515, 8.103573,  1.975449, 3.983412, 6.023890,
        8.096884};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 4>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, NoPack2) {
    ntt::tensor<float, ntt::fixed_shape<1, 13, 2>> buffer_1;
    ntt::tensor<float, ntt::fixed_shape<13, 2>> buffer_4;
    ntt::tensor<float, ntt::fixed_shape<13, 2>> buffer_7;
    std::iota(buffer_1.elements().begin(), buffer_1.elements().end(), 0.f);
    std::iota(buffer_4.elements().begin(), buffer_4.elements().end(), 0.f);
    std::iota(buffer_7.elements().begin(), buffer_7.elements().end(), 0.f);

    // no pack with pad
    ntt::tensor<float, ntt::fixed_shape<1, 13, 2>> ntt_output;
    packed_rms_norm<1>(buffer_1, buffer_4, buffer_7, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    const float array_golden[] = {
        0.000000,  1.068599,  2.274398,  3.617395,  5.097591,  6.714986,
        8.469580,  10.361372, 12.390364, 14.556555, 16.859943, 19.300531,
        21.878319, 24.593304, 27.445490, 30.434874, 33.561455, 36.825237,
        40.226219, 43.764397, 47.439774, 51.252350, 55.202126, 59.289101,
        63.513275, 67.874649};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 13, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, Pack0) {
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
    packed_rms_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> ntt_output;
    unpack<1>(buffer_6, ntt_output);

    const float array_golden[] = {
        0.000000,  1.055427,  2.221709,  3.498847,  4.886838,  6.385685,
        7.995387,  9.715942,  11.547354, 13.489619, 15.542740, 17.706715,
        19.981546, 22.367229, 24.863770, 27.471165, 30.189415, 33.018517,
        35.958477, 39.009293, 42.170959, 45.443481, 48.826862, 52.321095,
        55.926186, 59.642124, 63.468922, 67.406570, 71.455078, 75.614441,
        79.884659, 84.265732};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 2>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, Pack1) {

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
    packed_rms_norm<2>(input_packed, scale_packed, bias_packed, output_packed,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 2, 16>> ntt_output;
    unpack<2>(output_packed, ntt_output);

    const float array_golden[] = {
        15.000000, 14.113592, 13.454370, 13.022331, 12.817478, 12.839809,
        13.089325, 13.566026, 14.269911, 15.200981, 16.359236, 17.744677,
        19.357300, 21.197109, 23.264103, 25.558281, 15.000000, 14.709875,
        14.503265, 14.380171, 14.340590, 14.384525, 14.511974, 14.722939,
        15.017418, 15.395411, 15.856919, 16.401943, 17.030481, 17.742533,
        18.538101, 19.417183};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 2, 16>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, Pack2) {
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
    packed_rms_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 4>> ntt_output;
    unpack<1>(buffer_4, ntt_output);

    const float array_golden[] = {
        1.000000, 3.069045, 6.207134, 10.414268, 1.712697, 3.781742, 6.207135,
        8.988876, 1.836333, 3.881750, 6.136250,  8.599833, 1.885856, 3.919355,
        6.100497, 8.429281, 1.912426, 3.938904,  6.079436, 8.334021, 1.928977,
        3.950852, 6.065625, 8.273296, 1.940273,  3.958902, 6.055888, 8.231230,
        1.948472, 3.964691, 6.048659, 8.200375,  1.954692, 3.969053, 6.043082,
        8.176779, 1.959574, 3.972457, 6.038650,  8.158153, 1.963506, 3.975187,
        6.035044, 8.143075, 1.966741, 3.977425,  6.032052, 8.130621, 1.969450,
        3.979293, 6.029531, 8.120161, 1.971750,  3.980876, 6.027376, 8.111252,
        1.973729, 3.982234, 6.025515, 8.103573,  1.975449, 3.983412, 6.023890,
        8.096884};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 4>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, Pack3) {
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
    packed_rms_norm<2>(packed_input, scale, bias, packed_output,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> ntt_output;
    unpack<1>(packed_output, ntt_output);

    const float array_golden[] = {
        0.000000,  1.239046,  2.956183,  5.151411,  7.824731,  10.976143,
        14.605645, 18.713240, 0.000000,  1.767523,  3.705606,  5.814250,
        8.093454,  10.543219, 13.163545, 15.954431, 0.000000,  1.865838,
        3.833540,  5.903105,  8.074533,  10.347824, 12.722979, 15.199997,
        0.000000,  1.905952,  3.884380,  5.935284,  8.058664,  10.254520,
        12.522853, 14.863661, 0.000000,  1.927647,  3.911515,  5.951605,
        8.047915,  10.200447, 12.409199, 14.674172, 0.000000,  1.941224,
        3.928361,  5.961412,  8.040377,  10.165253, 12.336044, 14.552748,
        0.000000,  1.950516,  3.939829,  5.967937,  8.034843,  10.140546,
        12.285045, 14.468340, 0.000000,  1.957274,  3.948136,  5.972587,
        8.030626,  10.122253, 12.247470, 14.406275, 0.000000,  1.962409,
        3.954430,  5.976064,  8.027309,  10.108169, 12.218640, 14.358725,
        0.000000,  1.966442,  3.959363,  5.978761,  8.024637,  10.096991,
        12.195823, 14.321133, 0.000000,  1.969695,  3.963333,  5.980914,
        8.022438,  10.087905, 12.177316, 14.290668, 0.000000,  1.972373,
        3.966597,  5.982672,  8.020597,  10.080375, 12.162003, 14.265482,
        0.000000,  1.974616,  3.969327,  5.984134,  8.019035,  10.074032,
        12.149124, 14.244310, 0.000000,  1.976522,  3.971645,  5.985369,
        8.017693,  10.068617, 12.138141, 14.226266, 0.000000,  1.978163,
        3.973638,  5.986425,  8.016525,  10.063938, 12.128664, 14.210703,
        0.000000,  1.979589,  3.975368,  5.987340,  8.015503,  10.059858,
        12.120404, 14.197142};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 8>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(FixedShapeRMSNorm, Pack4) {
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
    packed_rms_norm<1>(packed_input, packed_scale, packed_bias, packed_output,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> ntt_output;
    unpack<2>(packed_output, ntt_output);

    const float array_golden[] = {
        0.000000,   1.013611,   2.054446,   3.122503,   4.217783,   5.340286,
        6.490011,   7.666960,   8.871131,   10.102526,  11.361143,  12.646983,
        13.960046,  15.300331,  16.667839,  18.062571,  19.484526,  20.933702,
        22.410103,  23.913725,  25.444571,  27.002640,  28.587931,  30.200445,
        31.840183,  33.507141,  35.201324,  36.922729,  38.671360,  40.447208,
        42.250282,  44.080582,  45.938103,  47.822845,  49.734810,  51.673996,
        53.640411,  55.634045,  57.654900,  59.702980,  61.778282,  63.880810,
        66.010559,  68.167526,  70.351723,  72.563141,  74.801781,  77.067642,
        79.360733,  81.681038,  84.028564,  86.403320,  88.805298,  91.234497,
        93.690918,  96.174568,  98.685440,  101.223526, 103.788834, 106.381378,
        109.001137, 111.648117, 114.322327, 117.023758, 119.752411, 122.508278,
        125.291374, 128.101700, 130.939240, 133.804016, 136.695984, 139.615204,
        142.561646, 145.535294, 148.536179, 151.564285, 154.619598, 157.702148,
        160.811920, 163.948914, 167.113129, 170.304565, 173.523239, 176.769119,
        180.042236, 183.342560, 186.670120, 190.024902, 193.406891, 196.816101,
        200.252563, 203.716217, 207.207123, 210.725235, 214.270569, 217.843140,
        221.442917, 225.069916, 228.724152, 232.405594, 236.114273, 239.850174,
        243.613297, 247.403625, 251.221191, 255.065979, 258.937988, 262.837219,
        266.763672, 270.717346, 274.698273, 278.706390, 282.741760, 286.804321,
        290.894104, 295.011108, 299.155334, 303.326813, 307.525513, 311.751404,
        316.004547, 320.284912, 324.592468, 328.927277, 333.289307, 337.678558,
        342.095032, 346.538696};

    auto ntt_golden =
        nncase::ntt::tload<ntt::tensor<float, ntt::fixed_shape<1, 16, 8>>,
                           float>(array_golden);

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, NoPack0) {

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
    packed_rms_norm<1>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    tensor_type1 ntt_golden(shape1);
    float array_golden[] = {
        0.000000,  1.055427,  2.221709,  3.498847,  4.886838,  6.385685,
        7.995387,  9.715942,  11.547354, 13.489619, 15.542740, 17.706715,
        19.981546, 22.367229, 24.863770, 27.471165, 30.189415, 33.018517,
        35.958477, 39.009293, 42.170959, 45.443481, 48.826862, 52.321095,
        55.926186, 59.642124, 63.468922, 67.406570, 71.455078, 75.614441,
        79.884659, 84.265732};
    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, NoPack1) {
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
    packed_rms_norm<2>(buffer_0, buffer_1, buffer_2, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        1.000000, 3.069045, 6.207134, 10.414268, 1.712697, 3.781742, 6.207135,
        8.988876, 1.836333, 3.881750, 6.136250,  8.599833, 1.885856, 3.919355,
        6.100497, 8.429281, 1.912426, 3.938904,  6.079436, 8.334021, 1.928977,
        3.950852, 6.065625, 8.273296, 1.940273,  3.958902, 6.055888, 8.231230,
        1.948472, 3.964691, 6.048659, 8.200375,  1.954692, 3.969053, 6.043082,
        8.176779, 1.959574, 3.972457, 6.038650,  8.158153, 1.963506, 3.975187,
        6.035044, 8.143075, 1.966741, 3.977425,  6.032052, 8.130621, 1.969450,
        3.979293, 6.029531, 8.120161, 1.971750,  3.980876, 6.027376, 8.111252,
        1.973729, 3.982234, 6.025515, 8.103573,  1.975449, 3.983412, 6.023890,
        8.096884};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, NoPack2) {

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
    packed_rms_norm<1>(buffer_1, buffer_4, buffer_7, ntt_output, 1e-06,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,  1.068599,  2.274398,  3.617395,  5.097591,  6.714986,
        8.469580,  10.361372, 12.390364, 14.556555, 16.859943, 19.300531,
        21.878319, 24.593304, 27.445490, 30.434874, 33.561455, 36.825237,
        40.226219, 43.764397, 47.439774, 51.252350, 55.202126, 59.289101,
        63.513275, 67.874649};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, Pack0) {
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
    packed_rms_norm<1>(buffer_3, buffer_4, buffer_5, buffer_6,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_6, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,  1.055427,  2.221709,  3.498847,  4.886838,  6.385685,
        7.995387,  9.715942,  11.547354, 13.489619, 15.542740, 17.706715,
        19.981546, 22.367229, 24.863770, 27.471165, 30.189415, 33.018517,
        35.958477, 39.009293, 42.170959, 45.443481, 48.826862, 52.321095,
        55.926186, 59.642124, 63.468922, 67.406570, 71.455078, 75.614441,
        79.884659, 84.265732};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, Pack1) {

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
    packed_rms_norm<2>(input_packed, scale_packed, bias_packed, output_packed,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<2>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<2>(output_packed, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        15.000000, 14.113592, 13.454370, 13.022331, 12.817478, 12.839809,
        13.089325, 13.566026, 14.269911, 15.200981, 16.359236, 17.744677,
        19.357300, 21.197109, 23.264103, 25.558281, 15.000000, 14.709875,
        14.503265, 14.380171, 14.340590, 14.384525, 14.511974, 14.722939,
        15.017418, 15.395411, 15.856919, 16.401943, 17.030481, 17.742533,
        18.538101, 19.417183};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, Pack2) {
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
    packed_rms_norm<2>(buffer_3, buffer_1, buffer_2, buffer_4,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(buffer_4, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        1.000000, 3.069045, 6.207134, 10.414268, 1.712697, 3.781742, 6.207135,
        8.988876, 1.836333, 3.881750, 6.136250,  8.599833, 1.885856, 3.919355,
        6.100497, 8.429281, 1.912426, 3.938904,  6.079436, 8.334021, 1.928977,
        3.950852, 6.065625, 8.273296, 1.940273,  3.958902, 6.055888, 8.231230,
        1.948472, 3.964691, 6.048659, 8.200375,  1.954692, 3.969053, 6.043082,
        8.176779, 1.959574, 3.972457, 6.038650,  8.158153, 1.963506, 3.975187,
        6.035044, 8.143075, 1.966741, 3.977425,  6.032052, 8.130621, 1.969450,
        3.979293, 6.029531, 8.120161, 1.971750,  3.980876, 6.027376, 8.111252,
        1.973729, 3.982234, 6.025515, 8.103573,  1.975449, 3.983412, 6.023890,
        8.096884};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, Pack3) {

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
    packed_rms_norm<2>(packed_input, scale, bias, packed_output,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<1>(packed_output, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,  1.239046,  2.956183,  5.151411,  7.824731,  10.976143,
        14.605645, 18.713240, 0.000000,  1.767523,  3.705606,  5.814250,
        8.093454,  10.543219, 13.163545, 15.954431, 0.000000,  1.865838,
        3.833540,  5.903105,  8.074533,  10.347824, 12.722979, 15.199997,
        0.000000,  1.905952,  3.884380,  5.935284,  8.058664,  10.254520,
        12.522853, 14.863661, 0.000000,  1.927647,  3.911515,  5.951605,
        8.047915,  10.200447, 12.409199, 14.674172, 0.000000,  1.941224,
        3.928361,  5.961412,  8.040377,  10.165253, 12.336044, 14.552748,
        0.000000,  1.950516,  3.939829,  5.967937,  8.034843,  10.140546,
        12.285045, 14.468340, 0.000000,  1.957274,  3.948136,  5.972587,
        8.030626,  10.122253, 12.247470, 14.406275, 0.000000,  1.962409,
        3.954430,  5.976064,  8.027309,  10.108169, 12.218640, 14.358725,
        0.000000,  1.966442,  3.959363,  5.978761,  8.024637,  10.096991,
        12.195823, 14.321133, 0.000000,  1.969695,  3.963333,  5.980914,
        8.022438,  10.087905, 12.177316, 14.290668, 0.000000,  1.972373,
        3.966597,  5.982672,  8.020597,  10.080375, 12.162003, 14.265482,
        0.000000,  1.974616,  3.969327,  5.984134,  8.019035,  10.074032,
        12.149124, 14.244310, 0.000000,  1.976522,  3.971645,  5.985369,
        8.017693,  10.068617, 12.138141, 14.226266, 0.000000,  1.978163,
        3.973638,  5.986425,  8.016525,  10.063938, 12.128664, 14.210703,
        0.000000,  1.979589,  3.975368,  5.987340,  8.015503,  10.059858,
        12.120404, 14.197142};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

TEST(RankedShapeRMSNorm, Pack4) {
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
    packed_rms_norm<1>(packed_input, packed_scale, packed_bias, packed_output,
                       ntt::vector<float, P>::from_scalar(1E-06),
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

    tensor_type1 ntt_output(shape1);
    unpack<2>(packed_output, ntt_output);

    tensor_type1 ntt_golden(shape1);
    const float array_golden[] = {
        0.000000,   1.013611,   2.054446,   3.122503,   4.217783,   5.340286,
        6.490011,   7.666960,   8.871131,   10.102526,  11.361143,  12.646983,
        13.960046,  15.300331,  16.667839,  18.062571,  19.484526,  20.933702,
        22.410103,  23.913725,  25.444571,  27.002640,  28.587931,  30.200445,
        31.840183,  33.507141,  35.201324,  36.922729,  38.671360,  40.447208,
        42.250282,  44.080582,  45.938103,  47.822845,  49.734810,  51.673996,
        53.640411,  55.634045,  57.654900,  59.702980,  61.778282,  63.880810,
        66.010559,  68.167526,  70.351723,  72.563141,  74.801781,  77.067642,
        79.360733,  81.681038,  84.028564,  86.403320,  88.805298,  91.234497,
        93.690918,  96.174568,  98.685440,  101.223526, 103.788834, 106.381378,
        109.001137, 111.648117, 114.322327, 117.023758, 119.752411, 122.508278,
        125.291374, 128.101700, 130.939240, 133.804016, 136.695984, 139.615204,
        142.561646, 145.535294, 148.536179, 151.564285, 154.619598, 157.702148,
        160.811920, 163.948914, 167.113129, 170.304565, 173.523239, 176.769119,
        180.042236, 183.342560, 186.670120, 190.024902, 193.406891, 196.816101,
        200.252563, 203.716217, 207.207123, 210.725235, 214.270569, 217.843140,
        221.442917, 225.069916, 228.724152, 232.405594, 236.114273, 239.850174,
        243.613297, 247.403625, 251.221191, 255.065979, 258.937988, 262.837219,
        266.763672, 270.717346, 274.698273, 278.706390, 282.741760, 286.804321,
        290.894104, 295.011108, 299.155334, 303.326813, 307.525513, 311.751404,
        316.004547, 320.284912, 324.592468, 328.927277, 333.289307, 337.678558,
        342.095032, 346.538696};

    size_t i = 0;
    ntt::apply(ntt_golden.shape(),
               [&](auto index) { ntt_golden(index) = array_golden[i++]; });

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
