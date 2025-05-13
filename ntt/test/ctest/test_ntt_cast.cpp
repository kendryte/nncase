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
#include "nncase/ntt/tensor.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/float8.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(CastTestFloat32ToInt32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<int32_t>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<int32_t>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32_ranked, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape = ntt::make_shape(M, N);

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<int32_t>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<int32_t>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<int32_t, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<int32_t>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<int32_t>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    int32_t min_input = -100;
    int32_t max_input = 100;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<int32_t>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    int32_t min_input = -100;
    int32_t max_input = 100;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<int32_t>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<int32_t, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = 0.f;
    float max_input = 100.0f;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<unsigned int>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<unsigned int>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = 0.f;
    float max_input = 100.0f;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<unsigned int, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<unsigned int>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<unsigned int>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<unsigned int>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<unsigned int>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<unsigned int, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<bool>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<bool>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool_1D, Pack) {
    constexpr dim_t N = 128;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape1 = ntt::fixed_shape_v<N>;
    auto shape2 = ntt::fixed_shape_v<N / P>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<bool, P * 4>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<bool>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<bool>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool_2D, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;
    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<bool, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<bool>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<bool>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    bool min_input = 0;
    bool max_input = 1;

    auto shape = ntt::fixed_shape_v<M, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<bool>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32_1D, Pack) {
    constexpr dim_t N = 128;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    bool min_input = 0;
    bool max_input = 1;

    auto shape1 = ntt::fixed_shape_v<N>;
    auto shape2 = ntt::fixed_shape_v<N / P>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<bool>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<bool, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32_2D, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    bool min_input = 0;
    bool max_input = 1;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<bool>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<bool, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToFloat8E4M3, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    float min_input = -500.0f;
    float max_input = 500.0f;

    auto shape = ntt::fixed_shape_v<M, N>;
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(shape);
    ntt::cast(ntt_input, ntt_output1);

    // float8
    alignas(32) auto ntt_output2 = ntt::make_tensor<float_e4m3_t>(shape);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = (float_e4m3_t)(ntt_input)(index);
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToFloat8E4M3, Pack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -500.0f;
    float max_input = 500.0f;

    auto shape1 = ntt::fixed_shape_v<M, N>;
    auto shape2 = ntt::fixed_shape_v<M / P, N>;

    // init
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(shape2);
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(shape1);
    ntt::unpack<0>(pack_output, ntt_output1);

    // float8
    alignas(32) auto ntt_output2 = ntt::make_tensor<float_e4m3_t>(shape1);
    nncase::ntt::apply(ntt_input.shape(), [&](auto index) {
        (ntt_output2)(index) = float_e4m3_t((ntt_input)(index));
    });

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
