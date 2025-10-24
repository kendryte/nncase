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
#include <bitset>
#include <gtest/gtest.h>
#include <nncase/bfloat16.h>
#include <nncase/float8.h>
#include <nncase/float_subbyte.h>
#include <nncase/half.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(CastFp4To, Float32_NoVectorize) {
    constexpr size_t M = 64;

    float_e2m1_t init_array[M] = {
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,
        1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,
        0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,
        -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,
        2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1,
        -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1};
    auto ntt_input = ntt::make_tensor_view(
        std::span<float_e2m1_t, M>(init_array, M), ntt::fixed_shape_v<1, M>);

    auto ntt_output_actual = ntt::make_tensor<float>(ntt::fixed_shape_v<1, M>);

    // ntt
    ntt::cast(ntt_input, ntt_output_actual, ntt::fixed_shape_v<>);

    float golden_array[] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    auto ntt_output_expected = ntt::make_tensor_view(
        std::span<float, M>(golden_array, M), ntt::fixed_shape_v<1, M>);

    EXPECT_TRUE(
        NttTest::compare_tensor(ntt_output_actual, ntt_output_expected));
}

TEST(CastFp4To, Float32_Vectorize) {
    constexpr size_t M = 64;
    constexpr size_t N = 2;
    constexpr size_t total_size = M * N;

    constexpr size_t PIn = NTT_VLEN / 4;
    constexpr size_t POut = NTT_VLEN / (sizeof(float) * 8);
    float_e2m1_t init_array[total_size] = {
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,
        1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,
        0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,
        -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,
        2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1,
        -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,
        1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,
        0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,
        -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,
        2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1,
        -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,
        1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,
        0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,
        -4_fe2m1, -6_fe2m1};

    auto ntt_input = ntt::make_tensor<float_e2m1_t>(ntt::fixed_shape_v<M, N>);

    size_t idx = 0;
    ntt::apply(ntt_input.shape(),
               [&](auto index) { ntt_input(index) = init_array[idx++]; });

    auto ntt_input_vectorized =
        ntt::make_tensor<ntt::vector<float_e2m1_t, PIn>>(
            ntt::fixed_shape_v<M / PIn, N>);
    ntt::pack(ntt_input, ntt_input_vectorized, ntt::fixed_shape_v<0>);

    auto ntt_output_vectorized = ntt::make_tensor<ntt::vector<float, POut>>(
        ntt::fixed_shape_v<M / POut, N>);
    auto ntt_output_actual = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);

    // ntt
    ntt::cast(ntt_input_vectorized, ntt_output_vectorized,
              ntt::fixed_shape_v<0>);

    ntt::unpack(ntt_output_vectorized, ntt_output_actual,
                ntt::fixed_shape_v<0>);

    auto ntt_output_expected =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    idx = 0;
    ntt::apply(ntt_output_expected.shape(), [&](auto index) {
        ntt_output_expected(index) = init_array[idx++];
    });

    EXPECT_TRUE(
        NttTest::compare_tensor(ntt_output_actual, ntt_output_expected));
}

TEST(CastFloat32To, Fp4_NoVectorize) {
    constexpr size_t M = 64;

    float init_array[M] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    auto ntt_input = ntt::make_tensor_view(std::span<float, M>(init_array, M),
                                           ntt::fixed_shape_v<1, M>);

    auto ntt_output_actual =
        ntt::make_tensor<float_e2m1_t>(ntt::fixed_shape_v<1, M>);

    // ntt
    ntt::cast(ntt_input, ntt_output_actual, ntt::fixed_shape_v<>);

    float_e2m1_t golden_array[] = {
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,
        1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,
        0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,
        -4_fe2m1, -6_fe2m1,   0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,
        2_fe2m1,  3_fe2m1,    4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1,
        -1_fe2m1, -1.5_fe2m1, -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1,
        0_fe2m1,  0.5_fe2m1,  1_fe2m1,  1.5_fe2m1,  2_fe2m1,  3_fe2m1,
        4_fe2m1,  6_fe2m1,    0_fe2m1,  -0.5_fe2m1, -1_fe2m1, -1.5_fe2m1,
        -2_fe2m1, -3_fe2m1,   -4_fe2m1, -6_fe2m1};
    auto ntt_output_expected = ntt::make_tensor_view(
        std::span<float_e2m1_t, M>(golden_array, M), ntt::fixed_shape_v<1, M>);

    EXPECT_TRUE(
        NttTest::compare_tensor(ntt_output_actual, ntt_output_expected));
}

TEST(CastFloat32To, Fp4_Vectorize) {
    constexpr size_t M = 64;
    constexpr size_t N = 2;
    constexpr size_t total_size = M * N;

    constexpr size_t PIn = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t POut = NTT_VLEN / 4;
    float init_array[total_size] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};

    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);

    size_t idx = 0;
    ntt::apply(ntt_input.shape(),
               [&](auto index) { ntt_input(index) = init_array[idx++]; });

    auto ntt_input_vectorized = ntt::make_tensor<ntt::vector<float, PIn>>(
        ntt::fixed_shape_v<M / PIn, N>);
    ntt::pack(ntt_input, ntt_input_vectorized, ntt::fixed_shape_v<0>);

    auto ntt_output_vectorized =
        ntt::make_tensor<ntt::vector<float_e2m1_t, POut>>(
            ntt::fixed_shape_v<M / POut, N>);
    auto ntt_output_actual =
        ntt::make_tensor<float_e2m1_t>(ntt::fixed_shape_v<M, N>);

    // ntt
    ntt::cast(ntt_input_vectorized, ntt_output_vectorized,
              ntt::fixed_shape_v<0>);

    ntt::unpack(ntt_output_vectorized, ntt_output_actual,
                ntt::fixed_shape_v<0>);

    auto ntt_output_expected =
        ntt::make_tensor<float_e2m1_t>(ntt::fixed_shape_v<M, N>);
    idx = 0;
    ntt::apply(ntt_output_expected.shape(), [&](auto index) {
        ntt_output_expected(index) = (float_e2m1_t)init_array[idx++];
    });

    EXPECT_TRUE(
        NttTest::compare_tensor(ntt_output_actual, ntt_output_expected));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
