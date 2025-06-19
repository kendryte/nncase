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

template <size_t N, size_t C, size_t H, size_t W, size_t P1, size_t P2,
          size_t P3, size_t P4, FixedDimensions TAxes, size_t... TransposePerms>
void fixed_unpack_test(const TAxes &axes) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t num_pack_dims = (P1 > 1 ? 1 : 0) + (P2 > 1 ? 1 : 0) +
                                     (P3 > 1 ? 1 : 0) + (P4 > 1 ? 1 : 0);

    using Elem =
        std::conditional_t<(num_pack_dims > 1), ntt::vector<float, P, P>,
                           ntt::vector<float, P>>;
    auto ntt_input = ntt::make_tensor<Elem>(
        ntt::fixed_shape_v<N / P1, C / P2, H / P3, W / P4>);
    NttTest::init_tensor(ntt_input, -10.0f, 10.0f);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    ntt::unpack(ntt_input, ntt_output1, axes);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {TransposePerms...};
    auto tmp = ortki_Transpose(ort_input, perms, sizeof...(TransposePerms));

    int64_t shape_data[] = {N, C, H, W};
    int64_t shape_dims[] = {4};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(shape_data), ort_type,
                             shape_dims, 1);
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, fixed_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<0>;

    fixed_unpack_test<N, C, H, W, P, 1, 1, 1, decltype(unpack_axes), 0, 4, 1, 2,
                      3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<1>;

    fixed_unpack_test<N, C, H, W, 1, P, 1, 1, decltype(unpack_axes), 0, 1, 4, 2,
                      3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_H_1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<2>;

    fixed_unpack_test<N, C, H, W, 1, 1, P, 1, decltype(unpack_axes), 0, 1, 2, 4,
                      3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_H_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = 2;

    auto unpack_axes = ntt::fixed_shape_v<2>;

    fixed_unpack_test<N, C, H, W, 1, 1, P, 1, decltype(unpack_axes), 0, 1, 2, 4,
                      3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;

    auto unpack_axes = ntt::fixed_shape_v<3>;

    fixed_unpack_test<N, C, H, W, 1, 1, 1, P, decltype(unpack_axes), 0, 1, 2, 3,
                      4>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_N_C_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<0, 1>;

    fixed_unpack_test<N, C, H, W, P, P, 1, 1, decltype(unpack_axes), 0, 4, 1, 5,
                      2, 3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_N_C_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P + 1;
    constexpr size_t W = P + 1;

    auto unpack_axes = ntt::fixed_shape_v<0, 1>;

    fixed_unpack_test<N, C, H, W, P, P, 1, 1, decltype(unpack_axes), 0, 4, 1, 5,
                      2, 3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_C_H_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<1, 2>;

    fixed_unpack_test<N, C, H, W, 1, P, P, 1, decltype(unpack_axes), 0, 1, 4, 2,
                      5, 3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_C_H_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2 + 1;

    auto unpack_axes = ntt::fixed_shape_v<1, 2>;

    fixed_unpack_test<N, C, H, W, 1, P, P, 1, decltype(unpack_axes), 0, 1, 4, 2,
                      5, 3>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_H_W_even) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2;

    auto unpack_axes = ntt::fixed_shape_v<2, 3>;

    fixed_unpack_test<N, C, H, W, 1, 1, P, P, decltype(unpack_axes), 0, 1, 2, 4,
                      3, 5>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_H_W_odd) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 7;
    constexpr size_t W = P * 5;

    auto unpack_axes = ntt::fixed_shape_v<2, 3>;

    fixed_unpack_test<N, C, H, W, 1, 1, P, P, decltype(unpack_axes), 0, 1, 2, 4,
                      3, 5>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_N_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;

    auto unpack_axes = ntt::fixed_shape_v<0, 3>;

    fixed_unpack_test<N, C, H, W, P, 1, 1, P, decltype(unpack_axes), 0, 4, 1, 2,
                      3, 5>(unpack_axes);
}

TEST(UnpackTestFloat, fixed_shape_dim_C_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<1, 0>;

    fixed_unpack_test<N, C, H, W, P, P, 1, 1, decltype(unpack_axes), 0, 5, 1, 4,
                      2, 3>(unpack_axes);
}

template <size_t N, size_t C, size_t H, size_t W, size_t P1, size_t P2,
          size_t P3, size_t P4, FixedDimensions TAxes, size_t... TransposePerms>
void ranked_unpack_test(const TAxes &axes) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t num_pack_dims = (P1 > 1 ? 1 : 0) + (P2 > 1 ? 1 : 0) +
                                     (P3 > 1 ? 1 : 0) + (P4 > 1 ? 1 : 0);

    using Elem =
        std::conditional_t<(num_pack_dims > 1), ntt::vector<float, P, P>,
                           ntt::vector<float, P>>;
    auto ntt_input =
        ntt::make_tensor<Elem>(ntt::make_shape(N / P1, C / P2, H / P3, W / P4));
    NttTest::init_tensor(ntt_input, -10.0f, 10.0f);

    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    ntt::unpack(ntt_input, ntt_output1, axes);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {TransposePerms...};
    auto tmp = ortki_Transpose(ort_input, perms, sizeof...(TransposePerms));

    int64_t shape_data[] = {N, C, H, W};
    int64_t shape_dims[] = {4};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape = make_tensor(reinterpret_cast<void *>(shape_data), ort_type,
                             shape_dims, 1);
    auto ort_output = ortki_Reshape(tmp, shape, 0);

    auto ntt_output2 = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(UnpackTestFloat, ranked_shape_dim_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<0>;

    ranked_unpack_test<N, C, H, W, P, 1, 1, 1, decltype(unpack_axes), 0, 4, 1,
                       2, 3>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<1>;

    ranked_unpack_test<N, C, H, W, 1, P, 1, 1, decltype(unpack_axes), 0, 1, 4,
                       2, 3>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<2>;

    ranked_unpack_test<N, C, H, W, 1, 1, P, 1, decltype(unpack_axes), 0, 1, 2,
                       4, 3>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P;
    constexpr size_t W = P * 2;

    auto unpack_axes = ntt::fixed_shape_v<3>;

    ranked_unpack_test<N, C, H, W, 1, 1, 1, P, decltype(unpack_axes), 0, 1, 2,
                       3, 4>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_N_C) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P * 2;
    constexpr size_t C = P * 2;
    constexpr size_t H = P;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<0, 1>;

    ranked_unpack_test<N, C, H, W, P, P, 1, 1, decltype(unpack_axes), 0, 4, 1,
                       5, 2, 3>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_C_H) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P * 2;
    constexpr size_t H = P * 2;
    constexpr size_t W = P;

    auto unpack_axes = ntt::fixed_shape_v<1, 2>;

    ranked_unpack_test<N, C, H, W, 1, P, P, 1, decltype(unpack_axes), 0, 1, 4,
                       2, 5, 3>(unpack_axes);
}

TEST(UnpackTestFloat, ranked_shape_dim_H_W) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t N = P;
    constexpr size_t C = P;
    constexpr size_t H = P * 2;
    constexpr size_t W = P * 2;

    auto unpack_axes = ntt::fixed_shape_v<2, 3>;

    ranked_unpack_test<N, C, H, W, 1, 1, P, P, decltype(unpack_axes), 0, 1, 2,
                       4, 3, 5>(unpack_axes);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}