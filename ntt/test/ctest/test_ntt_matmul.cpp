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
#include "nncase/bfloat16.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

#define MATMUL_INPUT_FLOAT_INIT                                                \
    auto ntt_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);        \
    auto ntt_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);        \
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);                                  \
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

#define NTT_MATMUL_FLOAT_COMPARE                                               \
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);                                  \
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);                                  \
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);                          \
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);    \
    NttTest::ort2ntt(ort_output, ntt_output2);                                 \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

TEST(MatmulTestFloat, NoVectorize) {
    // init
    MATMUL_INPUT_FLOAT_INIT

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    ntt::matmul<false>(ntt_lhs, ntt_rhs, ntt_output1, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>);

    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);

    // ort
    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_M) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    ntt::matmul<false>(p_ntt_lhs, ntt_rhs, tmp, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    ntt::matmul<false>(ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<1>);

    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_M_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<0, 1>);

    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_M_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32 / P, 32>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape_v<0, 1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<0>);

    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<32, 32 / P>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<1>);

    NTT_MATMUL_FLOAT_COMPARE
}

TEST(MatmulTestFloat, Vectorize_M_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_FLOAT_INIT

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::vectorize(ntt_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::vectorize(ntt_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<32, 32>);
    auto tmp = ntt::make_tensor<ntt::vector<float, P, P>>(
        ntt::fixed_shape_v<32 / P, 32 / P>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape_v<0, 1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0, 1>,
                       ntt::fixed_shape_v<>);
    devectorize(tmp, ntt_output1, ntt::fixed_shape_v<0, 1>);

    NTT_MATMUL_FLOAT_COMPARE
}

#define MATMUL_INPUT_INIT(type, shape, min_val, max_val)                       \
    auto ntt_f8_lhs = ntt::make_tensor<type>(shape);                           \
    auto ntt_f8_rhs = ntt::make_tensor<type>(shape);                           \
    NttTest::init_tensor(ntt_f8_lhs, (type)(min_val), (type)(max_val));        \
    NttTest::init_tensor(ntt_f8_rhs, (type)(min_val), (type)(max_val));

#define MATMUL_OUTPUT_COMPARE(shape, ntt_f8_lhs, ntt_f8_rhs, ntt_output)       \
    auto ntt_f32_lhs = ntt::make_tensor<float>(shape);                         \
    auto ntt_f32_rhs = ntt::make_tensor<float>(shape);                         \
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);                                        \
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);                                        \
                                                                               \
    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);                              \
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);                              \
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);                          \
                                                                               \
    auto ntt_output2 = ntt::make_tensor<float>(shape);                         \
    NttTest::ort2ntt(ort_output, ntt_output2);                                 \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));

TEST(MatmulTestFloatE4M3Float32, NoVectorize) {
    // init
    MATMUL_INPUT_INIT(float_e4m3_t, (ntt::fixed_shape_v<64, 64>), -448.f,
                      448.f);

    // ntt
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<64, 64>);
    ntt::matmul<false>(ntt_f8_lhs, ntt_f8_rhs, ntt_output);

    MATMUL_OUTPUT_COMPARE((ntt::fixed_shape_v<64, 64>), ntt_f8_lhs, ntt_f8_rhs,
                          ntt_output)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_K0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e4m3_t) * 8);

    // init
    MATMUL_INPUT_INIT(float_e4m3_t, (ntt::fixed_shape_v<128, 128>), -448.f,
                      448.f);

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128, 128 / P>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128 / P, 128>);
    ntt::vectorize(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);

    MATMUL_OUTPUT_COMPARE((ntt::fixed_shape_v<128, 128>), ntt_f8_lhs,
                          ntt_f8_rhs, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_K1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    MATMUL_INPUT_INIT(float_e4m3_t, (ntt::fixed_shape_v<128, 128>), -448.f,
                      448.f)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128, 128 / P>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128 / P, 128>);
    ntt::vectorize(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);

    MATMUL_OUTPUT_COMPARE((ntt::fixed_shape_v<128, 128>), ntt_f8_lhs,
                          ntt_f8_rhs, ntt_output1)
}

#define INIT_F8_MATMUL_TENSORS(M, K, N)                                        \
    auto ntt_lhs_f8 =                                                          \
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, K>);              \
    auto ntt_rhs_f8 =                                                          \
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<K, N>);              \
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t) - 448.f,                   \
                         (float_e4m3_t)448.f);                                 \
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t) - 448.f,                   \
                         (float_e4m3_t)448.f);

#define VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)                           \
    auto ntt_lhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);      \
    auto ntt_rhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);      \
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);                                        \
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);                                        \
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);                              \
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);                              \
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);                          \
                                                                               \
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);      \
    NttTest::ort2ntt(ort_output, ntt_output2);                                 \
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));

TEST(MatmulTestFloatE4M3Float32, Vectorize_M0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<K, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<K, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    p_ntt_lhs = ntt_lhs_f8;
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    p_ntt_lhs = ntt_lhs_f8;
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_K0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K / P1, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_K1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K / P1, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_f32 =
        ntt::make_tensor<ntt::vector<float, P2>>(ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_f32 = ntt::make_tensor<ntt::vector<float, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_f32 = ntt::make_tensor<ntt::vector<float, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_f32 = ntt::make_tensor<ntt::vector<float, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Float32, Vectorize_M_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_f32 = ntt::make_tensor<ntt::vector<float, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_f32, ntt_output1, ntt::fixed_shape_v<0, 1>);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output1)
}

TEST(MatmulTestFloatE4M3Bfloat16, NoVectorize) {
    // init
    MATMUL_INPUT_INIT(float_e4m3_t, (ntt::fixed_shape_v<64, 64>), -448.f,
                      448.f);

    // ntt
    auto ntt_output = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<64, 64>);
    ntt::matmul<false>(ntt_f8_lhs, ntt_f8_rhs, ntt_output);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<64, 64>);
    ntt::cast(ntt_output, ntt_output_f32);

    MATMUL_OUTPUT_COMPARE((ntt::fixed_shape_v<64, 64>), ntt_f8_lhs, ntt_f8_rhs,
                          ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_K0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e4m3_t) * 8);

    // init
    auto ntt_lhs_f8 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<128, 128>);
    auto ntt_rhs_f8 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<128, 128>);
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128, 128 / P>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128 / P, 128>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<128, 128>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::cast(ntt_output1, ntt_output_f32);

    auto ntt_lhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    auto ntt_rhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_K1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    auto ntt_lhs_f8 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<128, 128>);
    auto ntt_rhs_f8 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<128, 128>);
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128, 128 / P>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<128 / P, 128>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<128, 128>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape_v<1>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<0>,
                       ntt::fixed_shape_v<>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::cast(ntt_output1, ntt_output_f32);

    auto ntt_lhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    auto ntt_rhs_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<128, 128>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<K, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<K, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    p_ntt_lhs = ntt_lhs_f8;
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<M, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    p_ntt_lhs = ntt_lhs_f8;
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_K0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K / P1, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_K1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K / P1, N>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M / P2, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2>>(
        ntt::fixed_shape_v<M, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0, 1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<M / P1, K>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1>>(
        ntt::fixed_shape_v<K, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0, 1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0, 1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

TEST(MatmulTestFloatE4M3Bfloat16, Vectorize_M_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    INIT_F8_MATMUL_TENSORS(M, K, N)

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<M / P1, K / P1>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<float_e4m3_t, P1, P1>>(
        ntt::fixed_shape_v<K / P1, N / P1>);
    ntt::vectorize(ntt_lhs_f8, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::vectorize(ntt_rhs_f8, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output_out = ntt::make_tensor<ntt::vector<bfloat16, P2, P2>>(
        ntt::fixed_shape_v<M / P2, N / P2>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<M, N>);
    devectorize(ntt_output_out, ntt_output1, ntt::fixed_shape_v<0, 1>);
    auto ntt_output_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    VERIFY_MATMUL_WITH_ORT(M, K, N, ntt_output_f32)
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
