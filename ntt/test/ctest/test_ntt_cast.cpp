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

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<int32_t, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32_ranked, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto shape = ntt::make_ranked_shape(M, N);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<2>>;
    using tensor_type2 = ntt::tensor<int32_t, ntt::ranked_shape<2>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1(shape);
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) tensor_type2 ntt_output2(shape);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToInt32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<int32_t, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<int32_t, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    int32_t min_input = -100;
    int32_t max_input = 100;

    using tensor_type1 = ntt::tensor<int32_t, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestInt32ToFloat32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    int32_t min_input = -100;
    int32_t max_input = 100;

    using tensor_type1 = ntt::tensor<int32_t, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<int32_t, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = 0.f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<unsigned int, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToUint32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = 0.f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<unsigned int, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<unsigned int, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    using tensor_type1 = ntt::tensor<unsigned int, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestUint32ToFloat32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    unsigned int min_input = 0;
    unsigned int max_input = 100;

    using tensor_type1 = ntt::tensor<unsigned int, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<unsigned int, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToBool, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<bool, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    bool min_input = 0;
    bool max_input = 1;

    using tensor_type1 = ntt::tensor<bool, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestBoolToFloat32, Pack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    bool min_input = 0;
    bool max_input = 1;

    using tensor_type1 = ntt::tensor<bool, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<bool, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(CastTestFloat32ToFloat8E4M3, NoPack) {
    constexpr size_t M = 32;
    constexpr size_t N = 32;
    float min_input = -500.0f;
    float max_input = 500.0f;
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type2 ntt_output1;
    ntt::cast(ntt_input, ntt_output1);

    // float8
    alignas(32) tensor_type2 ntt_output2;
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

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, N>>;
    using tensor_type3 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;
    using tensor_type4 =
        ntt::tensor<ntt::vector<float_e4m3_t, P>, ntt::fixed_shape<M / P, N>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) tensor_type3 pack_input;
    alignas(32) tensor_type4 pack_output;
    ntt::pack<0>(ntt_input, pack_input);
    ntt::cast(pack_input, pack_output);
    alignas(32) tensor_type2 ntt_output1;
    ntt::unpack<0>(pack_output, ntt_output1);

    // float8
    alignas(32) tensor_type2 ntt_output2;
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
