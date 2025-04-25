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

TEST(MatmulTestFloat, NoPack) {
    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    // ntt
    tensor_type ntt_output1;
    ntt::matmul<false>(ntt_lhs, ntt_rhs, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_M) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_lhs;
    ntt::pack<0>(ntt_lhs, p_ntt_lhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, ntt_rhs, tmp, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{});
    unpack<0>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        tmp;
    ntt::matmul<false>(ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    unpack<1>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_rhs;
    ntt::pack<0>(ntt_lhs, p_ntt_lhs);
    ntt::pack<1>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    unpack<0, 1>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});
    unpack<0>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_rhs;
    ntt::pack<1>(ntt_lhs, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{});
    unpack<1>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    tensor_type ntt_lhs;
    tensor_type ntt_rhs;
    NttTest::init_tensor(ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(ntt_rhs, -2.f, 2.f);

    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs, p_ntt_rhs);

    // ntt
    tensor_type ntt_output1;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{});
    unpack<0, 1>(tmp, ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, NoPack) {
    // init
    using tensorA_F8_type = ntt::tensor<float_e4m3_t, ntt::fixed_shape<64, 64>>;
    using tensorB_F8_type = ntt::tensor<float_e4m3_t, ntt::fixed_shape<64, 64>>;
    using tensorA_F32_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    using tensorB_F32_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    using tensorC_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    tensorA_F8_type ntt_f8_lhs;
    tensorB_F8_type ntt_f8_rhs;
    NttTest::init_tensor(ntt_f8_lhs, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_f8_rhs, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    // ntt
    tensorC_type ntt_output;
    ntt::matmul<false>(ntt_f8_lhs, ntt_f8_rhs, ntt_output);

    tensorA_F32_type ntt_f32_lhs;
    tensorB_F32_type ntt_f32_rhs;
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // // compare
    tensorC_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_K0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e4m3_t) * 8);

    // init
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<128, 128>>;
    using tensor_type_f8 =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<128, 128>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8 ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128, 128 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128 / P, 128>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    tensor_type_f32 ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});

    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32 ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_K1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<128, 128>>;
    using tensor_type_f8 =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<128, 128>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8 ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128, 128 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128 / P, 128>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    tensor_type_f32 ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});

    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32 ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32) tensor_type_f8_rhs p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M / P2, N>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32) tensor_type_f8_rhs p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M / P2, N>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) tensor_type_f8_lhs p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    p_ntt_lhs = ntt_lhs_f8;
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) tensor_type_f8_lhs p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    p_ntt_lhs = ntt_lhs_f8;
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_K0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<M / P1, K / P1>>
        p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K / P1, N>>
            p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M / P2, N>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_K1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<M / P1, K / P1>>
        p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K / P1, N>>
            p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M / P2, N>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M, K / P1>>
            p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<K / P1, N / P1>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M, K / P1>>
            p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<K / P1, N / P1>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2>, ntt::fixed_shape<M, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2, P2>,
                            ntt::fixed_shape<M / P2, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0, 1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2, P2>,
                            ntt::fixed_shape<M / P2, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0, 1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_K_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<M / P1, K / P1>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<K / P1, N / P1>>
        p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2, P2>,
                            ntt::fixed_shape<M / P2, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0, 1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Float32, Pack_M_K_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<M / P1, K / P1>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P1, P1>,
                            ntt::fixed_shape<K / P1, N / P1>>
        p_ntt_rhs;
    ntt::pack<0, 1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0, 1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32) ntt::tensor<ntt::vector<float, P2, P2>,
                            ntt::fixed_shape<M / P2, N / P2>>
        ntt_output_f32;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_f32,
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0, 1>{}, ntt::fixed_shape<0>{});
    tensor_type_f32_out ntt_output1;
    unpack<0, 1>(ntt_output_f32, ntt_output1);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, NoPack) {
    // init
    using tensorA_F8_type = ntt::tensor<float_e4m3_t, ntt::fixed_shape<64, 64>>;
    using tensorB_F8_type = ntt::tensor<float_e4m3_t, ntt::fixed_shape<64, 64>>;
    using tensorA_F32_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    using tensorB_F32_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    using tensorC_F32_type = ntt::tensor<float, ntt::fixed_shape<64, 64>>;
    using tensorA_Out_type = ntt::tensor<bfloat16, ntt::fixed_shape<64, 64>>;
    using tensorB_Out_type = ntt::tensor<bfloat16, ntt::fixed_shape<64, 64>>;
    using tensorC_Out_type = ntt::tensor<bfloat16, ntt::fixed_shape<64, 64>>;
    tensorA_F8_type ntt_f8_lhs;
    tensorB_F8_type ntt_f8_rhs;
    NttTest::init_tensor(ntt_f8_lhs, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_f8_rhs, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    // ntt
    tensorC_Out_type ntt_output;
    ntt::matmul<false>(ntt_f8_lhs, ntt_f8_rhs, ntt_output);
    tensorC_F32_type ntt_output_f32;
    ntt::cast(ntt_output, ntt_output_f32);

    tensorA_F32_type ntt_f32_lhs;
    tensorB_F32_type ntt_f32_rhs;
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // // compare
    tensorC_F32_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_K0) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e4m3_t) * 8);

    // init
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<128, 128>>;
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<128, 128>>;
    using tensor_type_f8 =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<128, 128>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8 ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128, 128 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128 / P, 128>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    tensor_type_out ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});
    tensor_type_f32 ntt_output_f32;
    ntt::cast(ntt_output1, ntt_output_f32);

    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32 ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_K1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<128, 128>>;
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<128, 128>>;
    using tensor_type_f8 =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<128, 128>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8 ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128, 128 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float_e4m3_t, P>,
                            ntt::fixed_shape<128 / P, 128>>
        p_ntt_rhs;
    ntt::pack<1>(ntt_lhs_f8, p_ntt_lhs);
    ntt::pack<0>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    tensor_type_out ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});
    tensor_type_f32 ntt_output_f32;
    ntt::cast(ntt_output1, ntt_output_f32);

    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32 ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);

    // ort
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_M0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<M, N>>;
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32) tensor_type_f8_rhs p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    alignas(32)
        ntt::tensor<ntt::vector<bfloat16, P2>, ntt::fixed_shape<M / P2, N>>
            ntt_output_out;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{});
    tensor_type_out ntt_output1;
    unpack<0>(ntt_output_out, ntt_output1);
    tensor_type_f32_out ntt_output_f32;
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_M1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<M, N>>;
    using tensor_type_f32 = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8 = ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8 ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<M / P1, K>>
            p_ntt_lhs;
    alignas(32) tensor_type_f8_rhs p_ntt_rhs;
    ntt::pack<0>(ntt_lhs_f8, p_ntt_lhs);
    p_ntt_rhs = ntt_rhs_f8;

    // ntt
    alignas(32)
        ntt::tensor<ntt::vector<bfloat16, P2>, ntt::fixed_shape<M / P2, N>>
            ntt_output_out;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output_out,
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<>{}, ntt::fixed_shape<0>{});
    tensor_type_out ntt_output1;
    unpack<0>(ntt_output_out, ntt_output1);
    tensor_type_f32_out ntt_output_f32;
    ntt::cast(ntt_output1, ntt_output_f32);

    // ort
    tensor_type_f32 ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_N0) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float_e4m3_t) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) tensor_type_f8_lhs p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    p_ntt_lhs = ntt_lhs_f8;
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32)
        ntt::tensor<ntt::vector<bfloat16, P2>, ntt::fixed_shape<M, N / P2>>
            ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    tensor_type_out ntt_output1_unpack;
    unpack<1>(ntt_output1, ntt_output1_unpack);
    tensor_type_f32_out ntt_output_f32;
    ntt::cast(ntt_output1_unpack, ntt_output_f32);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

TEST(MatmulTestFloatE4M3Bfloat16, Pack_N1) {
    constexpr size_t P1 = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t P2 = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    // init
    using tensor_type_out = ntt::tensor<bfloat16, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_out = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type_f32_lhs = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    using tensor_type_f32_rhs = ntt::tensor<float, ntt::fixed_shape<K, N>>;
    using tensor_type_f8_lhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<M, K>>;
    using tensor_type_f8_rhs =
        ntt::tensor<float_e4m3_t, ntt::fixed_shape<K, N>>;
    tensor_type_f8_lhs ntt_lhs_f8;
    tensor_type_f8_rhs ntt_rhs_f8;
    NttTest::init_tensor(ntt_lhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);
    NttTest::init_tensor(ntt_rhs_f8, (float_e4m3_t)-448.f, (float_e4m3_t)448.f);

    alignas(32) tensor_type_f8_lhs p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float_e4m3_t, P1>, ntt::fixed_shape<K, N / P1>>
            p_ntt_rhs;
    p_ntt_lhs = ntt_lhs_f8;
    ntt::pack<1>(ntt_rhs_f8, p_ntt_rhs);

    // ntt
    alignas(32)
        ntt::tensor<ntt::vector<bfloat16, P2>, ntt::fixed_shape<M, N / P2>>
            ntt_output1;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    tensor_type_out ntt_output1_unpack;
    unpack<1>(ntt_output1, ntt_output1_unpack);
    tensor_type_f32_out ntt_output_f32;
    ntt::cast(ntt_output1_unpack, ntt_output_f32);

    // ort
    tensor_type_f32_lhs ntt_lhs_f32;
    tensor_type_f32_rhs ntt_rhs_f32;
    ntt::cast(ntt_lhs_f8, ntt_lhs_f32);
    ntt::cast(ntt_rhs_f8, ntt_rhs_f32);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs_f32);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs_f32);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    tensor_type_f32_out ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output_f32, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
