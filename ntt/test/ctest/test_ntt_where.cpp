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
#include <nncase/float8.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(WhereTestFloat, fixed_fixed_fixed_devectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 2;
    constexpr size_t w = 2;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_fixed_fixed_devectorize) {
    constexpr size_t n = 8;
    constexpr size_t c = 8;
    constexpr size_t h = 8;
    constexpr size_t w = 8;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_scalar_fixed_devectorize) {
    constexpr size_t n = 8;
    constexpr size_t c = 8;
    constexpr size_t h = 8;
    constexpr size_t w = 8;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_fixed_scalar_devectorize) {
    constexpr size_t n = 8;
    constexpr size_t c = 8;
    constexpr size_t h = 8;
    constexpr size_t w = 8;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_scalar_scalar_devectorize) {
    // constexpr size_t n = 32;
    // constexpr size_t c = 32;
    // constexpr size_t h = 32;
    // constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<16>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<16>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_fixed_scalar_devectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 8;
    constexpr size_t h = 8;
    constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_scalar_fixed_devectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 8;
    constexpr size_t h = 8;
    constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_scalar_scalar_devectorize) {
    // constexpr size_t n = 1;
    // constexpr size_t c = 8;
    // constexpr size_t h = 8;
    // constexpr size_t w = 32;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    // ntt
    ntt::where(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_fixed_fixed_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_input1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_input2_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto condition_vectorized = ntt::make_tensor<ntt::vector<bool, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    // ntt
    ntt::vectorize(ntt_input1, ntt_input1_vectorized, fixed_shape_v<3>);
    ntt::vectorize(ntt_input2, ntt_input2_vectorized, fixed_shape_v<3>);
    ntt::vectorize(condition, condition_vectorized, fixed_shape_v<3>);

    ntt::where(condition_vectorized, ntt_input1_vectorized, ntt_input2_vectorized,
               ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_fixed_scalar_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_input1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto condition_vectorized = ntt::make_tensor<ntt::vector<bool, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    // ntt
    ntt::vectorize(ntt_input1, ntt_input1_vectorized, fixed_shape_v<3>);
    ntt::vectorize(condition, condition_vectorized, fixed_shape_v<3>);

    ntt::where(condition_vectorized, ntt_input1_vectorized, ntt_input2,
               ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_scalar_fixed_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_input2_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto condition_vectorized = ntt::make_tensor<ntt::vector<bool, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    // ntt
    ntt::vectorize(ntt_input2, ntt_input2_vectorized, fixed_shape_v<3>);
    ntt::vectorize(condition, condition_vectorized, fixed_shape_v<3>);

    ntt::where(condition_vectorized, ntt_input1, ntt_input2_vectorized,
               ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_fixed_fixed_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_input1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_input2_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    // ntt
    ntt::vectorize(ntt_input1, ntt_input1_vectorized, fixed_shape_v<3>);
    ntt::vectorize(ntt_input2, ntt_input2_vectorized, fixed_shape_v<3>);

    ntt::where(condition, ntt_input1_vectorized, ntt_input2_vectorized,
               ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, fixed_scalar_scalar_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(condition, 0, 1);

    auto condition_vectorized = ntt::make_tensor<ntt::vector<bool, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);

    // ntt
    ntt::vectorize(condition, condition_vectorized, fixed_shape_v<3>);

    ntt::where(condition_vectorized, ntt_input1, ntt_input2, ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_scalar_fixed_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);
    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);

    auto ntt_input2_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);

    // ntt
    ntt::vectorize(ntt_input2, ntt_input2_vectorized, fixed_shape_v<3>);

    ntt::where(condition, ntt_input1, ntt_input2_vectorized, ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(WhereTestFloat, scalar_fixed_scalar_vectorize) {
    constexpr size_t n = 1;
    constexpr size_t c = 1;
    constexpr size_t h = 32;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    auto condition = ntt::make_tensor<bool>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(condition, 0, 1);

    auto ntt_input2 = ntt::make_tensor<float>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input2, min_input, max_input);
    auto ntt_input1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    NttTest::init_tensor(ntt_input1, min_input, max_input);

    auto ntt_input1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);
    auto ntt_output1_vectorized = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<n, c, h, w / P>);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);

    // ntt
    ntt::vectorize(ntt_input1, ntt_input1_vectorized, fixed_shape_v<3>);

    ntt::where(condition, ntt_input1_vectorized, ntt_input2, ntt_output1_vectorized);
    ntt::devectorize(ntt_output1_vectorized, ntt_output1, fixed_shape_v<3>);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
