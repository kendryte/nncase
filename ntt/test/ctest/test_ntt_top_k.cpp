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

TEST(top_k_1d, largest_sorted_axis_0) {

    constexpr int64_t axis = 0;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

TEST(top_k_2d, largest_sorted_axis_1) {

    constexpr int64_t axis = 1;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t N = 5;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, k>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, k>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, k>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, k>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

TEST(top_k_2d, largest_sorted_axis_0) {

    constexpr int64_t axis = 0;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t N = 5;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k, N>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k, N>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k, N>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k, N>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

TEST(top_k_3d, largest_sorted_axis_0) {

    constexpr int64_t axis = 0;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t K = 9;
    constexpr int64_t N = 5;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K, N>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k, K, N>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k, K, N>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<k, K, N>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<k, K, N>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

TEST(top_k_3d, largest_sorted_axis_1) {

    constexpr int64_t axis = 1;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t K = 9;
    constexpr int64_t N = 5;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K, N>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, k, N>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, k, N>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, k, N>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, k, N>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

TEST(top_k_3d, largest_sorted_axis_2) {

    constexpr int64_t axis = 2;
    const int64_t largest = 1;
    const int64_t sorted = 1;
    constexpr int64_t M = 7;
    constexpr int64_t K = 9;
    constexpr int64_t N = 5;
    constexpr int64_t k = 2;

    auto ntt_probs = ntt::make_tensor<float>(ntt::fixed_shape_v<M, K, N>);
    auto ntt_k = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_probs, -10.f, 10.f);
    ntt_k(0) = k;

    auto ntt_top_k_probs_expect =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, K, k>);
    auto ntt_top_k_indices_expect =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, K, k>);

    auto ntt_top_k_probs_actual =
        ntt::make_tensor<float>(ntt::fixed_shape_v<M, K, k>);
    auto ntt_top_k_indices_actual =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<M, K, k>);

    ntt::top_k(ntt_probs, ntt_k, ntt_top_k_probs_actual,
               ntt_top_k_indices_actual, fixed_dim_v<axis>, largest, sorted);

    auto ort_probs = NttTest::ntt2ort(ntt_probs);
    auto ort_k = NttTest::ntt2ort(ntt_k);
    auto ort_output = ortki_TopK(ort_probs, ort_k, axis, largest, sorted);
    auto ort_top_k_probs = tensor_seq_get_value(ort_output, 0);
    auto ort_top_k_idices = tensor_seq_get_value(ort_output, 1);
    NttTest::ort2ntt(ort_top_k_probs, ntt_top_k_probs_expect);
    NttTest::ort2ntt(ort_top_k_idices, ntt_top_k_indices_expect);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_top_k_probs_expect,
                                        ntt_top_k_probs_actual) &&
                NttTest::compare_tensor(ntt_top_k_indices_expect,
                                        ntt_top_k_indices_actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
