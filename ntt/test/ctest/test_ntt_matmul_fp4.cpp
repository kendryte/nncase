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
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(MatmulTestFloatE2M1Float32, Vectorize_K) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_M) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, ntt_f8_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_N) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(ntt_f8_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_MN) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_M_K_N) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_M_K) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Float32, Vectorize_K_N) {
    using TIn = float_e2m1_t;
    using TOut = float;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_K) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_M) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, ntt_f8_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_N) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(ntt_f8_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_MN) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_M_K_N) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_M_K) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1Half, Vectorize_K_N) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<TOut>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_K) {
    using TIn = float_e2m1_t;
    using TOut = half;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output1_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    ntt::cast(ntt_output1, ntt_output1_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_f32, ntt_output2));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_M) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, ntt_f8_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_N) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(ntt_f8_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_MN) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M / PIn, K>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_M_K_N) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut, POut>>(
        ntt::fixed_shape_v<M / POut, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0, 1>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_M_K) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<M / PIn, K / PIn>);
    auto p_ntt_rhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<K / PIn, N>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<0, 1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<0>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

TEST(MatmulTestFloatE2M1BFloat16, Vectorize_K_N) {
    using TIn = float_e2m1_t;
    using TOut = bfloat16;
    constexpr size_t PIn = NTT_VLEN / (element_size_in_byte_v<TIn> * 8);
    constexpr size_t POut = NTT_VLEN / (element_size_in_byte_v<TOut> * 8);
    constexpr size_t M = 64;
    constexpr size_t K = 64;
    constexpr size_t N = 64;
    TIn min_val = -6_fe2m1;
    TIn max_val = 6_fe2m1;
    auto ntt_f8_lhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<M, K>);
    auto ntt_f8_rhs = ntt::make_tensor<TIn>(ntt::fixed_shape_v<K, N>);
    NttTest::init_tensor(ntt_f8_lhs, (TIn)(min_val), (TIn)(max_val));
    NttTest::init_tensor(ntt_f8_rhs, (TIn)(min_val), (TIn)(max_val));

    auto p_ntt_lhs =
        ntt::make_tensor<ntt::vector<TIn, PIn>>(ntt::fixed_shape_v<M, K / PIn>);
    auto p_ntt_rhs = ntt::make_tensor<ntt::vector<TIn, PIn, PIn>>(
        ntt::fixed_shape_v<K / PIn, N / PIn>);
    ntt::pack(ntt_f8_lhs, p_ntt_lhs, ntt::fixed_shape_v<1>);
    ntt::pack(ntt_f8_rhs, p_ntt_rhs, ntt::fixed_shape_v<0, 1>);

    // ntt
    auto ntt_output1 = ntt::make_tensor<ntt::vector<TOut, POut>>(
        ntt::fixed_shape_v<M, N / POut>);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, ntt_output1, nullptr,
                       ntt::fixed_shape_v<1>, ntt::fixed_shape_v<>,
                       ntt::fixed_shape_v<0, 1>, ntt::fixed_shape_v<>);
    auto ntt_output2 = ntt::make_tensor<TOut>(ntt::fixed_shape_v<M, N>);
    auto ntt_output2_f32 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    unpack(ntt_output1, ntt_output2, ntt::fixed_shape_v<1>);

    auto ntt_f32_lhs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K>);
    auto ntt_f32_rhs = ntt::make_tensor<float>(ntt::fixed_shape_v<K, N>);
    ntt::cast(ntt_f8_lhs, ntt_f32_lhs);
    ntt::cast(ntt_f8_rhs, ntt_f32_rhs);

    auto ort_lhs = NttTest::ntt2ort(ntt_f32_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_f32_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    auto ntt_output3 = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::ort2ntt(ort_output, ntt_output3);
    ntt::cast(ntt_output2, ntt_output2_f32);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output2_f32, ntt_output3));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
