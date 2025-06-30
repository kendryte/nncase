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
#include "nncase/ntt/shape.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <nncase/bfloat16.h>
#include <nncase/float8.h>
#include <nncase/half.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(CastTestFloat32ToInt32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    auto ntt_output2 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32_ranked, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape = ntt::make_shape(M, N);

    // init
    auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<int32_t>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    auto ntt_output2 = ntt::make_tensor<int32_t>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<int32_t, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    auto ntt_output2 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    int32_t min_input = -100;
    int32_t max_input = 100;

    // init
    auto ntt_input = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    int32_t min_input = -100;
    int32_t max_input = 100;

    // init
    auto ntt_input = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<int32_t, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = 0.f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    auto ntt_output2 = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = 0.f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output = ntt::make_tensor<ntt::vector<unsigned int, P>>(
        ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    auto ntt_output2 = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    // init
    auto ntt_input = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    // init
    auto ntt_input = ntt::make_tensor<unsigned int>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input = ntt::make_tensor<ntt::vector<unsigned int, P>>(
        ntt::fixed_shape_v<M / P, N>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    auto ntt_output2 = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool_1D, Pack) {
    constexpr size_t N = 128;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P>);
    auto pack_output = ntt::make_tensor<ntt::vector<bool, P * 4>>(
        ntt::fixed_shape_v<N / P / 4>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    auto ntt_output2 = ntt::make_tensor<bool>(ntt::fixed_shape_v<N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool_2D, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    auto ntt_output2 = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    bool min_input = 0;
    bool max_input = 1;

    // init
    auto ntt_input = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32_1D, Pack) {
    constexpr size_t N = 128;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    bool min_input = 0;
    bool max_input = 1;

    // init
    auto ntt_input = ntt::make_tensor<bool>(ntt::fixed_shape_v<N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<N / P>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<N / P>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32_2D, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    bool min_input = 0;
    bool max_input = 1;

    // init
    auto ntt_input = ntt::make_tensor<bool>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToFloat8E4M3, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -500.0f;
    float max_input = 500.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // float8
    auto ntt_output2 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (float_e4m3_t)(ntt_input)(index);
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToFloat8E4M3, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -500.0f;
    float max_input = 500.0f;

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    auto pack_output = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<M / P, N>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::cast(pack_input, pack_output);
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // float8
    auto ntt_output2 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = float_e4m3_t((ntt_input)(index));
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float_e4m3_t min_input = (float_e4m3_t)-448.0f;
    float_e4m3_t max_input = (float_e4m3_t)448.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // float8
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (float)(ntt_input)(index);
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToFloat32, Pack) {
    constexpr size_t M = 64;
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);
    float_e4m3_t min_input = (float_e4m3_t)-500.0f;
    float_e4m3_t max_input = (float_e4m3_t)500.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    auto pack_input = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);

    auto pack_output =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M / P2>);
    ntt::cast(pack_input, pack_output);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (float)((ntt_input)(index));
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToBFloat16, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float_e4m3_t min_input = (float_e4m3_t)-448.0f;
    float_e4m3_t max_input = (float_e4m3_t)448.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // float8
    auto ntt_output2 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (bfloat16)(ntt_input)(index);
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToBFloat16, Pack) {
    constexpr size_t M = 64;
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(bfloat16) * 8);
    float_e4m3_t min_input = (float_e4m3_t)-500.0f;
    float_e4m3_t max_input = (float_e4m3_t)500.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    auto pack_input = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);

    auto pack_output =
        ntt::make_tensor<ntt::vector<bfloat16, P2>>(ntt::fixed_shape_v<M / P2>);
    ntt::cast(pack_input, pack_output);

    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    auto ntt_output2 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (bfloat16)((ntt_input)(index));
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToHalf, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float_e4m3_t min_input = (float_e4m3_t)-448.0f;
    float_e4m3_t max_input = (float_e4m3_t)448.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_input, ntt_output1);

    // float8
    auto ntt_output2 = ntt::make_tensor<half>(ntt::fixed_shape_v<M, N>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (half)(ntt_input)(index);
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat8E4M3ToHalf, Pack) {
    constexpr size_t M = 64;
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(half) * 8);
    float_e4m3_t min_input = (float_e4m3_t)-500.0f;
    float_e4m3_t max_input = (float_e4m3_t)500.0f;

    // init
    auto ntt_input = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    auto pack_input = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1>);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);

    auto pack_output =
        ntt::make_tensor<ntt::vector<half, P2>>(ntt::fixed_shape_v<M / P2>);
    ntt::cast(pack_input, pack_output);

    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<M>);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    auto ntt_output2 = ntt::make_tensor<half>(ntt::fixed_shape_v<M>);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (half)((ntt_input)(index));
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
