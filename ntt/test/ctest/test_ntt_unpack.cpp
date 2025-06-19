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

TEST(UnpackTestFloat, fixed_shape_rest) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t M = P * 32;
    constexpr size_t N = 7;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M / P, N>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 2, 1};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {M, N};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<N / P, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<N, C / P, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_H_1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_H_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<N, C, H / P, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<N, C, H, W / P>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(ort_input, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_N_C_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_N_C_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P + 1;
    constexpr size_t W = P + 1;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N / P, C / P, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_C_H_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_C_H_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2 + 1;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N, C / P, H / P, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_H_W_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_H_W_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 7;
    constexpr size_t W = P * 5;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N, C, H / P, W / P>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_N_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N / P, C, H, W / P>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 2, 3, 5};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_C_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<N / P, C, H, W / P>);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 2, 3, 5};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(N / P, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(N, C / P, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(N, C, H / P, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(N, C, H, W / P));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(ort_input, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_N_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::make_shape(N / P, C / P, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_C_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::make_shape(N, C / P, H / P, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_H_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2;
    float min_input = -10.0f;
    float max_input = 10.0f;

    // init
    auto ntt_input = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::make_shape(N, C, H / P, W / P));
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto tmp = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t data[] = {N, C, H, W};
    int64_t data_shape[] = {std::size(data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,
                             data_shape, std::size(data_shape));
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}