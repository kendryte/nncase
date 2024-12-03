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

template <size_t N, size_t C, size_t H, size_t W, size_t perm_n, size_t perm_c,
          size_t perm_h, size_t perm_w>
void fixed_shape_transpose() {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr std::array<size_t, 4> org_dims = {N, C, H, W};
    constexpr std::array<size_t, 4> new_dims = {
        org_dims[perm_n], org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]};

    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<N, C, H, W>>;
    using tensor_type2 = ntt::tensor<
        ntt::vector<float, P>,
        ntt::fixed_shape<new_dims[0], new_dims[1], new_dims[2], new_dims[3]>>;

    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, 4};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

template <size_t N, size_t C, size_t H, size_t W, size_t perm_n, size_t perm_c,
          size_t perm_h, size_t perm_w>
void ranked_shape_transpose() {

    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr std::array<size_t, 4> org_dims = {N, C, H, W};
    constexpr std::array<size_t, 4> new_dims = {
        org_dims[perm_n], org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]};

    auto shape1 = ntt::make_ranked_shape(N, C, H, W);
    auto shape2 = ntt::make_ranked_shape(new_dims[0], new_dims[1], new_dims[2],
                                         new_dims[3]);
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<4>>;

    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, 4};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(FixedShapeTranspose4D, NCHW) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 1, 2, 3>();
}
TEST(FixedShapeTranspose4D, NCWH) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 1, 3, 2>();
}
TEST(FixedShapeTranspose4D, NHCW) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 2, 1, 3>();
}
TEST(FixedShapeTranspose4D, NHWC) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 2, 3, 1>();
}
TEST(FixedShapeTranspose4D, NWCH) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 3, 1, 2>();
}
TEST(FixedShapeTranspose4D, NWHC) {
    fixed_shape_transpose<16, 16, 16, 16, 0, 3, 2, 1>();
}
TEST(FixedShapeTranspose4D, CNHW) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 0, 2, 3>();
}
TEST(FixedShapeTranspose4D, CNWH) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 0, 3, 2>();
}
TEST(FixedShapeTranspose4D, CHNW) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 2, 0, 3>();
}
TEST(FixedShapeTranspose4D, CHWN) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 2, 3, 0>();
}
TEST(FixedShapeTranspose4D, CWNH) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 3, 0, 2>();
}
TEST(FixedShapeTranspose4D, CWHN) {
    fixed_shape_transpose<16, 16, 16, 16, 1, 3, 2, 0>();
}
TEST(FixedShapeTranspose4D, HNCW) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 0, 1, 3>();
}
TEST(FixedShapeTranspose4D, HNWC) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 0, 3, 1>();
}
TEST(FixedShapeTranspose4D, HCNW) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 1, 0, 3>();
}
TEST(FixedShapeTranspose4D, HCWN) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 1, 3, 0>();
}
TEST(FixedShapeTranspose4D, HWNC) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 3, 0, 1>();
}
TEST(FixedShapeTranspose4D, HWCN) {
    fixed_shape_transpose<16, 16, 16, 16, 2, 3, 1, 0>();
}
TEST(FixedShapeTranspose4D, WNCH) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 0, 1, 2>();
}
TEST(FixedShapeTranspose4D, WNHC) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 0, 2, 1>();
}
TEST(FixedShapeTranspose4D, WCNH) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 1, 0, 2>();
}
TEST(FixedShapeTranspose4D, WCHN) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 1, 2, 0>();
}
TEST(FixedShapeTranspose4D, WHNC) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 2, 0, 1>();
}
TEST(FixedShapeTranspose4D, WHCN) {
    fixed_shape_transpose<16, 16, 16, 16, 3, 2, 1, 0>();
}

TEST(RankedShapeTranspose4D, NCHW) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 1, 2, 3>();
}
TEST(RankedShapeTranspose4D, NCWH) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 1, 3, 2>();
}
TEST(RankedShapeTranspose4D, NHCW) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 2, 1, 3>();
}
TEST(RankedShapeTranspose4D, NHWC) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 2, 3, 1>();
}
TEST(RankedShapeTranspose4D, NWCH) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 3, 1, 2>();
}
TEST(RankedShapeTranspose4D, NWHC) {
    ranked_shape_transpose<16, 16, 16, 16, 0, 3, 2, 1>();
}
TEST(RankedShapeTranspose4D, CNHW) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 0, 2, 3>();
}
TEST(RankedShapeTranspose4D, CNWH) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 0, 3, 2>();
}
TEST(RankedShapeTranspose4D, CHNW) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 2, 0, 3>();
}
TEST(RankedShapeTranspose4D, CHWN) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 2, 3, 0>();
}
TEST(RankedShapeTranspose4D, CWNH) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 3, 0, 2>();
}
TEST(RankedShapeTranspose4D, CWHN) {
    ranked_shape_transpose<16, 16, 16, 16, 1, 3, 2, 0>();
}
TEST(RankedShapeTranspose4D, HNCW) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 0, 1, 3>();
}
TEST(RankedShapeTranspose4D, HNWC) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 0, 3, 1>();
}
TEST(RankedShapeTranspose4D, HCNW) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 1, 0, 3>();
}
TEST(RankedShapeTranspose4D, HCWN) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 1, 3, 0>();
}
TEST(RankedShapeTranspose4D, HWNC) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 3, 0, 1>();
}
TEST(RankedShapeTranspose4D, HWCN) {
    ranked_shape_transpose<16, 16, 16, 16, 2, 3, 1, 0>();
}
TEST(RankedShapeTranspose4D, WNCH) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 0, 1, 2>();
}
TEST(RankedShapeTranspose4D, WNHC) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 0, 2, 1>();
}
TEST(RankedShapeTranspose4D, WCNH) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 1, 0, 2>();
}
TEST(RankedShapeTranspose4D, WCHN) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 1, 2, 0>();
}
TEST(RankedShapeTranspose4D, WHNC) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 2, 0, 1>();
}
TEST(RankedShapeTranspose4D, WHCN) {
    ranked_shape_transpose<16, 16, 16, 16, 3, 2, 1, 0>();
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
