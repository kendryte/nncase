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

template <size_t perm_h, size_t perm_w>
void transpose_2D_fixed_shape_devectorized() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {h, w};

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_h], org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1, ntt::fixed_shape_v<perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_h], org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DFixedShapeUnVectorized, HW) {
    transpose_2D_fixed_shape_devectorized<0, 1>();
}

TEST(Transpose2DFixedShapeUnVectorized, WH) {
    transpose_2D_fixed_shape_devectorized<1, 0>();
}

template <size_t perm_h, size_t perm_w>
void transpose_2D_ranked_shape_devectorized() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {h, w};

    // init
    auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(h, w));
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_h], org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1, ntt::fixed_shape_v<perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_h], org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DRankedShapeUnVectorized, HW) {
    transpose_2D_ranked_shape_devectorized<0, 1>();
}

TEST(Transpose2DRankedShapeUnVectorized, WH) {
    transpose_2D_ranked_shape_devectorized<1, 0>();
}

template <size_t perm_h, size_t perm_w> void transpose_2D_fixed_shape_vectorized() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {h, w};

    // init
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<h, w>);
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_h], org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1, ntt::fixed_shape_v<perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_h], org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DFixedShapeVectorized, HW) {
    transpose_2D_fixed_shape_vectorized<0, 1>();
}

TEST(Transpose2DFixedShapeVectorized, WH) {
    transpose_2D_fixed_shape_vectorized<1, 0>();
}

template <size_t perm_h, size_t perm_w>
void transpose_2D_ranked_shape_vectorized() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {h, w};

    // init
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(h, w));
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_h], org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1, ntt::fixed_shape_v<perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_h], org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DRankedShapeVectorized, HW) {
    transpose_2D_ranked_shape_vectorized<0, 1>();
}

TEST(Transpose2DRankedShapeVectorized, WH) {
    transpose_2D_ranked_shape_vectorized<1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_fixed_shape_devectorized() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {c, h, w};

    // ntt
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<c, h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DFixedShapeUnVectorized, CHW) {
    transpose_3D_fixed_shape_devectorized<0, 1, 2>();
}

TEST(Transpose3DFixedShapeUnVectorized, CWH) {
    transpose_3D_fixed_shape_devectorized<0, 2, 1>();
}

TEST(Transpose3DFixedShapeUnVectorized, HCW) {
    transpose_3D_fixed_shape_devectorized<1, 0, 2>();
}

TEST(Transpose3DFixedShapeUnVectorized, HWC) {
    transpose_3D_fixed_shape_devectorized<1, 2, 0>();
}

TEST(Transpose3DFixedShapeUnVectorized, WCH) {
    transpose_3D_fixed_shape_devectorized<2, 0, 1>();
}

TEST(Transpose3DFixedShapeUnVectorized, WHC) {
    transpose_3D_fixed_shape_devectorized<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_ranked_shape_devectorized() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {c, h, w};

    // ntt
    auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(c, h, w));
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DRankedShapeUnVectorized, CHW) {
    transpose_3D_ranked_shape_devectorized<0, 1, 2>();
}

TEST(Transpose3DRankedShapeUnVectorized, CWH) {
    transpose_3D_ranked_shape_devectorized<0, 2, 1>();
}

TEST(Transpose3DRankedShapeUnVectorized, HCW) {
    transpose_3D_ranked_shape_devectorized<1, 0, 2>();
}

TEST(Transpose3DRankedShapeUnVectorized, HWC) {
    transpose_3D_ranked_shape_devectorized<1, 2, 0>();
}

TEST(Transpose3DRankedShapeUnVectorized, WCH) {
    transpose_3D_ranked_shape_devectorized<2, 0, 1>();
}

TEST(Transpose3DRankedShapeUnVectorized, WHC) {
    transpose_3D_fixed_shape_devectorized<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_fixed_shape_vectorized() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {c, h, w};

    // ntt
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<c, h, w>);
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DFixedShapeVectorized, CHW) {
    transpose_3D_fixed_shape_vectorized<0, 1, 2>();
}

TEST(Transpose3DFixedShapeVectorized, CWH) {
    transpose_3D_fixed_shape_vectorized<0, 2, 1>();
}

TEST(Transpose3DFixedShapeVectorized, HCW) {
    transpose_3D_fixed_shape_vectorized<1, 0, 2>();
}

TEST(Transpose3DFixedShapeVectorized, HWC) {
    transpose_3D_fixed_shape_vectorized<1, 2, 0>();
}

TEST(Transpose3DFixedShapeVectorized, WCH) {
    transpose_3D_fixed_shape_vectorized<2, 0, 1>();
}

TEST(Transpose3DFixedShapeVectorized, WHC) {
    transpose_3D_fixed_shape_vectorized<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_ranked_shape_vectorized() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {c, h, w};

    // ntt
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(c, h, w));
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DRankedShapeVectorized, CHW) {
    transpose_3D_ranked_shape_vectorized<0, 1, 2>();
}

TEST(Transpose3DRankedShapeVectorized, CWH) {
    transpose_3D_ranked_shape_vectorized<0, 2, 1>();
}

TEST(Transpose3DRankedShapeVectorized, HCW) {
    transpose_3D_ranked_shape_vectorized<1, 0, 2>();
}

TEST(Transpose3DRankedShapeVectorized, HWC) {
    transpose_3D_ranked_shape_vectorized<1, 2, 0>();
}

TEST(Transpose3DRankedShapeVectorized, WCH) {
    transpose_3D_ranked_shape_vectorized<2, 0, 1>();
}

TEST(Transpose3DRankedShapeVectorized, WHC) {
    transpose_3D_ranked_shape_vectorized<2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_fixed_shape_devectorized() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {n, c, h, w};

    // ntt
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::fixed_shape_v<org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DFixedShapeUnVectorized, NCHW) {
    transpose_4D_fixed_shape_devectorized<0, 1, 2, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, NCWH) {
    transpose_4D_fixed_shape_devectorized<0, 1, 3, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, NHCW) {
    transpose_4D_fixed_shape_devectorized<0, 2, 1, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, NHWC) {
    transpose_4D_fixed_shape_devectorized<0, 2, 3, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, NWCH) {
    transpose_4D_fixed_shape_devectorized<0, 3, 1, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, NWHC) {
    transpose_4D_fixed_shape_devectorized<0, 3, 2, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, CNHW) {
    transpose_4D_fixed_shape_devectorized<1, 0, 2, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, CNWH) {
    transpose_4D_fixed_shape_devectorized<1, 0, 3, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, CHNW) {
    transpose_4D_fixed_shape_devectorized<1, 2, 0, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, CHWN) {
    transpose_4D_fixed_shape_devectorized<1, 2, 3, 0>();
}

TEST(Transpose4DFixedShapeUnVectorized, CWNH) {
    transpose_4D_fixed_shape_devectorized<1, 3, 0, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, CWHN) {
    transpose_4D_fixed_shape_devectorized<1, 3, 2, 0>();
}

TEST(Transpose4DFixedShapeUnVectorized, HNCW) {
    transpose_4D_fixed_shape_devectorized<2, 0, 1, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, HNWC) {
    transpose_4D_fixed_shape_devectorized<2, 0, 3, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, HCNW) {
    transpose_4D_fixed_shape_devectorized<2, 1, 0, 3>();
}

TEST(Transpose4DFixedShapeUnVectorized, HCWN) {
    transpose_4D_fixed_shape_devectorized<2, 1, 3, 0>();
}

TEST(Transpose4DFixedShapeUnVectorized, HWNC) {
    transpose_4D_fixed_shape_devectorized<2, 3, 0, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, HWCN) {
    transpose_4D_fixed_shape_devectorized<2, 3, 1, 0>();
}

TEST(Transpose4DFixedShapeUnVectorized, WNCH) {
    transpose_4D_fixed_shape_devectorized<3, 0, 1, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, WNHC) {
    transpose_4D_fixed_shape_devectorized<3, 0, 2, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, WCNH) {
    transpose_4D_fixed_shape_devectorized<3, 1, 0, 2>();
}

TEST(Transpose4DFixedShapeUnVectorized, WCHN) {
    transpose_4D_fixed_shape_devectorized<3, 1, 2, 0>();
}

TEST(Transpose4DFixedShapeUnVectorized, WHNC) {
    transpose_4D_fixed_shape_devectorized<3, 2, 0, 1>();
}

TEST(Transpose4DFixedShapeUnVectorized, WHCN) {
    transpose_4D_fixed_shape_devectorized<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_ranked_shape_devectorized() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {n, c, h, w};

    // ntt
    auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(n, c, h, w));
    auto ntt_output1 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                        org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<float>(
        ntt::make_shape(org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                        org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DRankedShapeUnVectorized, NCHW) {
    transpose_4D_ranked_shape_devectorized<0, 1, 2, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, NCWH) {
    transpose_4D_ranked_shape_devectorized<0, 1, 3, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, NHCW) {
    transpose_4D_ranked_shape_devectorized<0, 2, 1, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, NHWC) {
    transpose_4D_ranked_shape_devectorized<0, 2, 3, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, NWCH) {
    transpose_4D_ranked_shape_devectorized<0, 3, 1, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, NWHC) {
    transpose_4D_ranked_shape_devectorized<0, 3, 2, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, CNHW) {
    transpose_4D_ranked_shape_devectorized<1, 0, 2, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, CNWH) {
    transpose_4D_ranked_shape_devectorized<1, 0, 3, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, CHNW) {
    transpose_4D_ranked_shape_devectorized<1, 2, 0, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, CHWN) {
    transpose_4D_ranked_shape_devectorized<1, 2, 3, 0>();
}

TEST(Transpose4DRankedShapeUnVectorized, CWNH) {
    transpose_4D_ranked_shape_devectorized<1, 3, 0, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, CWHN) {
    transpose_4D_ranked_shape_devectorized<1, 3, 2, 0>();
}

TEST(Transpose4DRankedShapeUnVectorized, HNCW) {
    transpose_4D_ranked_shape_devectorized<2, 0, 1, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, HNWC) {
    transpose_4D_ranked_shape_devectorized<2, 0, 3, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, HCNW) {
    transpose_4D_ranked_shape_devectorized<2, 1, 0, 3>();
}

TEST(Transpose4DRankedShapeUnVectorized, HCWN) {
    transpose_4D_ranked_shape_devectorized<2, 1, 3, 0>();
}

TEST(Transpose4DRankedShapeUnVectorized, HWNC) {
    transpose_4D_ranked_shape_devectorized<2, 3, 0, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, HWCN) {
    transpose_4D_ranked_shape_devectorized<2, 3, 1, 0>();
}

TEST(Transpose4DRankedShapeUnVectorized, WNCH) {
    transpose_4D_ranked_shape_devectorized<3, 0, 1, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, WNHC) {
    transpose_4D_ranked_shape_devectorized<3, 0, 2, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, WCNH) {
    transpose_4D_ranked_shape_devectorized<3, 1, 0, 2>();
}

TEST(Transpose4DRankedShapeUnVectorized, WCHN) {
    transpose_4D_ranked_shape_devectorized<3, 1, 2, 0>();
}

TEST(Transpose4DRankedShapeUnVectorized, WHNC) {
    transpose_4D_ranked_shape_devectorized<3, 2, 0, 1>();
}

TEST(Transpose4DRankedShapeUnVectorized, WHCN) {
    transpose_4D_ranked_shape_devectorized<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_fixed_shape_vectorized() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {n, c, h, w};

    // ntt
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<n, c, h, w>);
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                           org_dims[perm_w]>);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DFixedShapeVectorized, NCHW) {
    transpose_4D_fixed_shape_vectorized<0, 1, 2, 3>();
}

TEST(Transpose4DFixedShapeVectorized, NCWH) {
    transpose_4D_fixed_shape_vectorized<0, 1, 3, 2>();
}

TEST(Transpose4DFixedShapeVectorized, NHCW) {
    transpose_4D_fixed_shape_vectorized<0, 2, 1, 3>();
}

TEST(Transpose4DFixedShapeVectorized, NHWC) {
    transpose_4D_fixed_shape_vectorized<0, 2, 3, 1>();
}

TEST(Transpose4DFixedShapeVectorized, NWCH) {
    transpose_4D_fixed_shape_vectorized<0, 3, 1, 2>();
}

TEST(Transpose4DFixedShapeVectorized, NWHC) {
    transpose_4D_fixed_shape_vectorized<0, 3, 2, 1>();
}

TEST(Transpose4DFixedShapeVectorized, CNHW) {
    transpose_4D_fixed_shape_vectorized<1, 0, 2, 3>();
}

TEST(Transpose4DFixedShapeVectorized, CNWH) {
    transpose_4D_fixed_shape_vectorized<1, 0, 3, 2>();
}

TEST(Transpose4DFixedShapeVectorized, CHNW) {
    transpose_4D_fixed_shape_vectorized<1, 2, 0, 3>();
}

TEST(Transpose4DFixedShapeVectorized, CHWN) {
    transpose_4D_fixed_shape_vectorized<1, 2, 3, 0>();
}

TEST(Transpose4DFixedShapeVectorized, CWNH) {
    transpose_4D_fixed_shape_vectorized<1, 3, 0, 2>();
}

TEST(Transpose4DFixedShapeVectorized, CWHN) {
    transpose_4D_fixed_shape_vectorized<1, 3, 2, 0>();
}

TEST(Transpose4DFixedShapeVectorized, HNCW) {
    transpose_4D_fixed_shape_vectorized<2, 0, 1, 3>();
}

TEST(Transpose4DFixedShapeVectorized, HNWC) {
    transpose_4D_fixed_shape_vectorized<2, 0, 3, 1>();
}

TEST(Transpose4DFixedShapeVectorized, HCNW) {
    transpose_4D_fixed_shape_vectorized<2, 1, 0, 3>();
}

TEST(Transpose4DFixedShapeVectorized, HCWN) {
    transpose_4D_fixed_shape_vectorized<2, 1, 3, 0>();
}

TEST(Transpose4DFixedShapeVectorized, HWNC) {
    transpose_4D_fixed_shape_vectorized<2, 3, 0, 1>();
}

TEST(Transpose4DFixedShapeVectorized, HWCN) {
    transpose_4D_fixed_shape_vectorized<2, 3, 1, 0>();
}

TEST(Transpose4DFixedShapeVectorized, WNCH) {
    transpose_4D_fixed_shape_vectorized<3, 0, 1, 2>();
}

TEST(Transpose4DFixedShapeVectorized, WNHC) {
    transpose_4D_fixed_shape_vectorized<3, 0, 2, 1>();
}

TEST(Transpose4DFixedShapeVectorized, WCNH) {
    transpose_4D_fixed_shape_vectorized<3, 1, 0, 2>();
}

TEST(Transpose4DFixedShapeVectorized, WCHN) {
    transpose_4D_fixed_shape_vectorized<3, 1, 2, 0>();
}

TEST(Transpose4DFixedShapeVectorized, WHNC) {
    transpose_4D_fixed_shape_vectorized<3, 2, 0, 1>();
}

TEST(Transpose4DFixedShapeVectorized, WHCN) {
    transpose_4D_fixed_shape_vectorized<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_ranked_shape_vectorized() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {n, c, h, w};

    // ntt
    auto ntt_input =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::make_shape(n, c, h, w));
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                        org_dims[perm_w]));
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose(ntt_input, ntt_output1,
                   ntt::fixed_shape_v<perm_n, perm_c, perm_h, perm_w>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    auto ntt_output2 = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::make_shape(org_dims[perm_n], org_dims[perm_c], org_dims[perm_h],
                        org_dims[perm_w]));
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DRankedShapeVectorized, NCHW) {
    transpose_4D_ranked_shape_vectorized<0, 1, 2, 3>();
}

TEST(Transpose4DRankedShapeVectorized, NCWH) {
    transpose_4D_ranked_shape_vectorized<0, 1, 3, 2>();
}

TEST(Transpose4DRankedShapeVectorized, NHCW) {
    transpose_4D_ranked_shape_vectorized<0, 2, 1, 3>();
}

TEST(Transpose4DRankedShapeVectorized, NHWC) {
    transpose_4D_ranked_shape_vectorized<0, 2, 3, 1>();
}

TEST(Transpose4DRankedShapeVectorized, NWCH) {
    transpose_4D_ranked_shape_vectorized<0, 3, 1, 2>();
}

TEST(Transpose4DRankedShapeVectorized, NWHC) {
    transpose_4D_ranked_shape_vectorized<0, 3, 2, 1>();
}

TEST(Transpose4DRankedShapeVectorized, CNHW) {
    transpose_4D_ranked_shape_vectorized<1, 0, 2, 3>();
}

TEST(Transpose4DRankedShapeVectorized, CNWH) {
    transpose_4D_ranked_shape_vectorized<1, 0, 3, 2>();
}

TEST(Transpose4DRankedShapeVectorized, CHNW) {
    transpose_4D_ranked_shape_vectorized<1, 2, 0, 3>();
}

TEST(Transpose4DRankedShapeVectorized, CHWN) {
    transpose_4D_ranked_shape_vectorized<1, 2, 3, 0>();
}

TEST(Transpose4DRankedShapeVectorized, CWNH) {
    transpose_4D_ranked_shape_vectorized<1, 3, 0, 2>();
}

TEST(Transpose4DRankedShapeVectorized, CWHN) {
    transpose_4D_ranked_shape_vectorized<1, 3, 2, 0>();
}

TEST(Transpose4DRankedShapeVectorized, HNCW) {
    transpose_4D_ranked_shape_vectorized<2, 0, 1, 3>();
}

TEST(Transpose4DRankedShapeVectorized, HNWC) {
    transpose_4D_ranked_shape_vectorized<2, 0, 3, 1>();
}

TEST(Transpose4DRankedShapeVectorized, HCNW) {
    transpose_4D_ranked_shape_vectorized<2, 1, 0, 3>();
}

TEST(Transpose4DRankedShapeVectorized, HCWN) {
    transpose_4D_ranked_shape_vectorized<2, 1, 3, 0>();
}

TEST(Transpose4DRankedShapeVectorized, HWNC) {
    transpose_4D_ranked_shape_vectorized<2, 3, 0, 1>();
}

TEST(Transpose4DRankedShapeVectorized, HWCN) {
    transpose_4D_ranked_shape_vectorized<2, 3, 1, 0>();
}

TEST(Transpose4DRankedShapeVectorized, WNCH) {
    transpose_4D_ranked_shape_vectorized<3, 0, 1, 2>();
}

TEST(Transpose4DRankedShapeVectorized, WNHC) {
    transpose_4D_ranked_shape_vectorized<3, 0, 2, 1>();
}

TEST(Transpose4DRankedShapeVectorized, WCNH) {
    transpose_4D_ranked_shape_vectorized<3, 1, 0, 2>();
}

TEST(Transpose4DRankedShapeVectorized, WCHN) {
    transpose_4D_ranked_shape_vectorized<3, 1, 2, 0>();
}

TEST(Transpose4DRankedShapeVectorized, WHNC) {
    transpose_4D_ranked_shape_vectorized<3, 2, 0, 1>();
}

TEST(Transpose4DRankedShapeVectorized, WHCN) {
    transpose_4D_ranked_shape_vectorized<3, 2, 1, 0>();
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
