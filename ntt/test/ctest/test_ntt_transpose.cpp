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
void transpose_2D_fixed_shape_unpacked() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {h, w};
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<h, w>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<org_dims[perm_h], org_dims[perm_w]>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_h, perm_w>>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DFixedShapeUnPacked, HW) {
    transpose_2D_fixed_shape_unpacked<0, 1>();
}

TEST(Transpose2DFixedShapeUnPacked, WH) {
    transpose_2D_fixed_shape_unpacked<1, 0>();
}

template <size_t perm_h, size_t perm_w>
void transpose_2D_ranked_shape_unpacked() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_h], org_dims[perm_w]);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<ndims>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_h, perm_w>>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DRankedShapeUnPacked, HW) {
    transpose_2D_ranked_shape_unpacked<0, 1>();
}

TEST(Transpose2DRankedShapeUnPacked, WH) {
    transpose_2D_ranked_shape_unpacked<1, 0>();
}

template <size_t perm_h, size_t perm_w> void transpose_2D_fixed_shape_packed() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {h, w};
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<h, w>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>,
                    ntt::fixed_shape<org_dims[perm_h], org_dims[perm_w]>>;

    // init
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_h, perm_w>>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DFixedShapePacked, HW) {
    transpose_2D_fixed_shape_packed<0, 1>();
}

TEST(Transpose2DFixedShapePacked, WH) {
    transpose_2D_fixed_shape_packed<1, 0>();
}

template <size_t perm_h, size_t perm_w>
void transpose_2D_ranked_shape_packed() {
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_h], org_dims[perm_w]);
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_h, perm_w>>(ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_h, perm_w, ndims};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose2DRankedShapePacked, HW) {
    transpose_2D_ranked_shape_packed<0, 1>();
}

TEST(Transpose2DRankedShapePacked, WH) {
    transpose_2D_ranked_shape_packed<1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_fixed_shape_unpacked() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {c, h, w};
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<c, h, w>>;
    using tensor_type2 =
        ntt::tensor<float, ntt::fixed_shape<org_dims[perm_c], org_dims[perm_h],
                                            org_dims[perm_w]>>;

    // ntt
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_c, perm_h, perm_w>>(ntt_input,
                                                             ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DFixedShapeUnPacked, CHW) {
    transpose_3D_fixed_shape_unpacked<0, 1, 2>();
}

TEST(Transpose3DFixedShapeUnPacked, CWH) {
    transpose_3D_fixed_shape_unpacked<0, 2, 1>();
}

TEST(Transpose3DFixedShapeUnPacked, HCW) {
    transpose_3D_fixed_shape_unpacked<1, 0, 2>();
}

TEST(Transpose3DFixedShapeUnPacked, HWC) {
    transpose_3D_fixed_shape_unpacked<1, 2, 0>();
}

TEST(Transpose3DFixedShapeUnPacked, WCH) {
    transpose_3D_fixed_shape_unpacked<2, 0, 1>();
}

TEST(Transpose3DFixedShapeUnPacked, WHC) {
    transpose_3D_fixed_shape_unpacked<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_ranked_shape_unpacked() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {c, h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(c, h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_c], org_dims[perm_h],
                                         org_dims[perm_w]);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<ndims>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_c, perm_h, perm_w>>(ntt_input,
                                                             ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DRankedShapeUnPacked, CHW) {
    transpose_3D_ranked_shape_unpacked<0, 1, 2>();
}

TEST(Transpose3DRankedShapeUnPacked, CWH) {
    transpose_3D_ranked_shape_unpacked<0, 2, 1>();
}

TEST(Transpose3DRankedShapeUnPacked, HCW) {
    transpose_3D_ranked_shape_unpacked<1, 0, 2>();
}

TEST(Transpose3DRankedShapeUnPacked, HWC) {
    transpose_3D_ranked_shape_unpacked<1, 2, 0>();
}

TEST(Transpose3DRankedShapeUnPacked, WCH) {
    transpose_3D_ranked_shape_unpacked<2, 0, 1>();
}

TEST(Transpose3DRankedShapeUnPacked, WHC) {
    transpose_3D_fixed_shape_unpacked<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_fixed_shape_packed() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {c, h, w};
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<c, h, w>>;
    using tensor_type2 = ntt::tensor<
        ntt::vector<float, P>,
        ntt::fixed_shape<org_dims[perm_c], org_dims[perm_h], org_dims[perm_w]>>;

    // ntt
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_c, perm_h, perm_w>>(ntt_input,
                                                             ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DFixedShapePacked, CHW) {
    transpose_3D_fixed_shape_packed<0, 1, 2>();
}

TEST(Transpose3DFixedShapePacked, CWH) {
    transpose_3D_fixed_shape_packed<0, 2, 1>();
}

TEST(Transpose3DFixedShapePacked, HCW) {
    transpose_3D_fixed_shape_packed<1, 0, 2>();
}

TEST(Transpose3DFixedShapePacked, HWC) {
    transpose_3D_fixed_shape_packed<1, 2, 0>();
}

TEST(Transpose3DFixedShapePacked, WCH) {
    transpose_3D_fixed_shape_packed<2, 0, 1>();
}

TEST(Transpose3DFixedShapePacked, WHC) {
    transpose_3D_fixed_shape_packed<2, 1, 0>();
}

template <size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_3D_ranked_shape_packed() {
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {c, h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(c, h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_c], org_dims[perm_h],
                                         org_dims[perm_w]);
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_c, perm_h, perm_w>>(ntt_input,
                                                             ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_c, perm_h, perm_w, ndims};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose3DRankedShapePacked, CHW) {
    transpose_3D_ranked_shape_packed<0, 1, 2>();
}

TEST(Transpose3DRankedShapePacked, CWH) {
    transpose_3D_ranked_shape_packed<0, 2, 1>();
}

TEST(Transpose3DRankedShapePacked, HCW) {
    transpose_3D_ranked_shape_packed<1, 0, 2>();
}

TEST(Transpose3DRankedShapePacked, HWC) {
    transpose_3D_ranked_shape_packed<1, 2, 0>();
}

TEST(Transpose3DRankedShapePacked, WCH) {
    transpose_3D_ranked_shape_packed<2, 0, 1>();
}

TEST(Transpose3DRankedShapePacked, WHC) {
    transpose_3D_ranked_shape_packed<2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_fixed_shape_unpacked() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {n, c, h, w};
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 =
        ntt::tensor<float,
                    ntt::fixed_shape<org_dims[perm_n], org_dims[perm_c],
                                     org_dims[perm_h], org_dims[perm_w]>>;

    // ntt
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DFixedShapeUnPacked, NCHW) {
    transpose_4D_fixed_shape_unpacked<0, 1, 2, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, NCWH) {
    transpose_4D_fixed_shape_unpacked<0, 1, 3, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, NHCW) {
    transpose_4D_fixed_shape_unpacked<0, 2, 1, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, NHWC) {
    transpose_4D_fixed_shape_unpacked<0, 2, 3, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, NWCH) {
    transpose_4D_fixed_shape_unpacked<0, 3, 1, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, NWHC) {
    transpose_4D_fixed_shape_unpacked<0, 3, 2, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, CNHW) {
    transpose_4D_fixed_shape_unpacked<1, 0, 2, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, CNWH) {
    transpose_4D_fixed_shape_unpacked<1, 0, 3, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, CHNW) {
    transpose_4D_fixed_shape_unpacked<1, 2, 0, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, CHWN) {
    transpose_4D_fixed_shape_unpacked<1, 2, 3, 0>();
}

TEST(Transpose4DFixedShapeUnPacked, CWNH) {
    transpose_4D_fixed_shape_unpacked<1, 3, 0, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, CWHN) {
    transpose_4D_fixed_shape_unpacked<1, 3, 2, 0>();
}

TEST(Transpose4DFixedShapeUnPacked, HNCW) {
    transpose_4D_fixed_shape_unpacked<2, 0, 1, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, HNWC) {
    transpose_4D_fixed_shape_unpacked<2, 0, 3, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, HCNW) {
    transpose_4D_fixed_shape_unpacked<2, 1, 0, 3>();
}

TEST(Transpose4DFixedShapeUnPacked, HCWN) {
    transpose_4D_fixed_shape_unpacked<2, 1, 3, 0>();
}

TEST(Transpose4DFixedShapeUnPacked, HWNC) {
    transpose_4D_fixed_shape_unpacked<2, 3, 0, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, HWCN) {
    transpose_4D_fixed_shape_unpacked<2, 3, 1, 0>();
}

TEST(Transpose4DFixedShapeUnPacked, WNCH) {
    transpose_4D_fixed_shape_unpacked<3, 0, 1, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, WNHC) {
    transpose_4D_fixed_shape_unpacked<3, 0, 2, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, WCNH) {
    transpose_4D_fixed_shape_unpacked<3, 1, 0, 2>();
}

TEST(Transpose4DFixedShapeUnPacked, WCHN) {
    transpose_4D_fixed_shape_unpacked<3, 1, 2, 0>();
}

TEST(Transpose4DFixedShapeUnPacked, WHNC) {
    transpose_4D_fixed_shape_unpacked<3, 2, 0, 1>();
}

TEST(Transpose4DFixedShapeUnPacked, WHCN) {
    transpose_4D_fixed_shape_unpacked<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_ranked_shape_unpacked() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t org_dims[] = {n, c, h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(n, c, h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_n], org_dims[perm_c],
                                         org_dims[perm_h], org_dims[perm_w]);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<ndims>>;
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DRankedShapeUnPacked, NCHW) {
    transpose_4D_ranked_shape_unpacked<0, 1, 2, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, NCWH) {
    transpose_4D_ranked_shape_unpacked<0, 1, 3, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, NHCW) {
    transpose_4D_ranked_shape_unpacked<0, 2, 1, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, NHWC) {
    transpose_4D_ranked_shape_unpacked<0, 2, 3, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, NWCH) {
    transpose_4D_ranked_shape_unpacked<0, 3, 1, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, NWHC) {
    transpose_4D_ranked_shape_unpacked<0, 3, 2, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, CNHW) {
    transpose_4D_ranked_shape_unpacked<1, 0, 2, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, CNWH) {
    transpose_4D_ranked_shape_unpacked<1, 0, 3, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, CHNW) {
    transpose_4D_ranked_shape_unpacked<1, 2, 0, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, CHWN) {
    transpose_4D_ranked_shape_unpacked<1, 2, 3, 0>();
}

TEST(Transpose4DRankedShapeUnPacked, CWNH) {
    transpose_4D_ranked_shape_unpacked<1, 3, 0, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, CWHN) {
    transpose_4D_ranked_shape_unpacked<1, 3, 2, 0>();
}

TEST(Transpose4DRankedShapeUnPacked, HNCW) {
    transpose_4D_ranked_shape_unpacked<2, 0, 1, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, HNWC) {
    transpose_4D_ranked_shape_unpacked<2, 0, 3, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, HCNW) {
    transpose_4D_ranked_shape_unpacked<2, 1, 0, 3>();
}

TEST(Transpose4DRankedShapeUnPacked, HCWN) {
    transpose_4D_ranked_shape_unpacked<2, 1, 3, 0>();
}

TEST(Transpose4DRankedShapeUnPacked, HWNC) {
    transpose_4D_ranked_shape_unpacked<2, 3, 0, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, HWCN) {
    transpose_4D_ranked_shape_unpacked<2, 3, 1, 0>();
}

TEST(Transpose4DRankedShapeUnPacked, WNCH) {
    transpose_4D_ranked_shape_unpacked<3, 0, 1, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, WNHC) {
    transpose_4D_ranked_shape_unpacked<3, 0, 2, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, WCNH) {
    transpose_4D_ranked_shape_unpacked<3, 1, 0, 2>();
}

TEST(Transpose4DRankedShapeUnPacked, WCHN) {
    transpose_4D_ranked_shape_unpacked<3, 1, 2, 0>();
}

TEST(Transpose4DRankedShapeUnPacked, WHNC) {
    transpose_4D_ranked_shape_unpacked<3, 2, 0, 1>();
}

TEST(Transpose4DRankedShapeUnPacked, WHCN) {
    transpose_4D_ranked_shape_unpacked<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_fixed_shape_packed() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {n, c, h, w};
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<n, c, h, w>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>,
                    ntt::fixed_shape<org_dims[perm_n], org_dims[perm_c],
                                     org_dims[perm_h], org_dims[perm_w]>>;

    // ntt
    alignas(32) tensor_type1 ntt_input;
    alignas(32) tensor_type2 ntt_output1;
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, std::size(org_dims)};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DFixedShapePacked, NCHW) {
    transpose_4D_fixed_shape_packed<0, 1, 2, 3>();
}

TEST(Transpose4DFixedShapePacked, NCWH) {
    transpose_4D_fixed_shape_packed<0, 1, 3, 2>();
}

TEST(Transpose4DFixedShapePacked, NHCW) {
    transpose_4D_fixed_shape_packed<0, 2, 1, 3>();
}

TEST(Transpose4DFixedShapePacked, NHWC) {
    transpose_4D_fixed_shape_packed<0, 2, 3, 1>();
}

TEST(Transpose4DFixedShapePacked, NWCH) {
    transpose_4D_fixed_shape_packed<0, 3, 1, 2>();
}

TEST(Transpose4DFixedShapePacked, NWHC) {
    transpose_4D_fixed_shape_packed<0, 3, 2, 1>();
}

TEST(Transpose4DFixedShapePacked, CNHW) {
    transpose_4D_fixed_shape_packed<1, 0, 2, 3>();
}

TEST(Transpose4DFixedShapePacked, CNWH) {
    transpose_4D_fixed_shape_packed<1, 0, 3, 2>();
}

TEST(Transpose4DFixedShapePacked, CHNW) {
    transpose_4D_fixed_shape_packed<1, 2, 0, 3>();
}

TEST(Transpose4DFixedShapePacked, CHWN) {
    transpose_4D_fixed_shape_packed<1, 2, 3, 0>();
}

TEST(Transpose4DFixedShapePacked, CWNH) {
    transpose_4D_fixed_shape_packed<1, 3, 0, 2>();
}

TEST(Transpose4DFixedShapePacked, CWHN) {
    transpose_4D_fixed_shape_packed<1, 3, 2, 0>();
}

TEST(Transpose4DFixedShapePacked, HNCW) {
    transpose_4D_fixed_shape_packed<2, 0, 1, 3>();
}

TEST(Transpose4DFixedShapePacked, HNWC) {
    transpose_4D_fixed_shape_packed<2, 0, 3, 1>();
}

TEST(Transpose4DFixedShapePacked, HCNW) {
    transpose_4D_fixed_shape_packed<2, 1, 0, 3>();
}

TEST(Transpose4DFixedShapePacked, HCWN) {
    transpose_4D_fixed_shape_packed<2, 1, 3, 0>();
}

TEST(Transpose4DFixedShapePacked, HWNC) {
    transpose_4D_fixed_shape_packed<2, 3, 0, 1>();
}

TEST(Transpose4DFixedShapePacked, HWCN) {
    transpose_4D_fixed_shape_packed<2, 3, 1, 0>();
}

TEST(Transpose4DFixedShapePacked, WNCH) {
    transpose_4D_fixed_shape_packed<3, 0, 1, 2>();
}

TEST(Transpose4DFixedShapePacked, WNHC) {
    transpose_4D_fixed_shape_packed<3, 0, 2, 1>();
}

TEST(Transpose4DFixedShapePacked, WCNH) {
    transpose_4D_fixed_shape_packed<3, 1, 0, 2>();
}

TEST(Transpose4DFixedShapePacked, WCHN) {
    transpose_4D_fixed_shape_packed<3, 1, 2, 0>();
}

TEST(Transpose4DFixedShapePacked, WHNC) {
    transpose_4D_fixed_shape_packed<3, 2, 0, 1>();
}

TEST(Transpose4DFixedShapePacked, WHCN) {
    transpose_4D_fixed_shape_packed<3, 2, 1, 0>();
}

template <size_t perm_n, size_t perm_c, size_t perm_h, size_t perm_w>
void transpose_4D_ranked_shape_packed() {
    constexpr size_t n = 4;
    constexpr size_t c = 3;
    constexpr size_t h = 16;
    constexpr size_t w = 32;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    constexpr size_t org_dims[] = {n, c, h, w};
    constexpr size_t ndims = std::size(org_dims);
    auto shape1 = ntt::make_ranked_shape(n, c, h, w);
    auto shape2 = ntt::make_ranked_shape(org_dims[perm_n], org_dims[perm_c],
                                         org_dims[perm_h], org_dims[perm_w]);
    using tensor_type1 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::ranked_shape<ndims>>;

    // init
    alignas(32) tensor_type1 ntt_input(shape1);
    alignas(32) tensor_type2 ntt_output1(shape2);
    NttTest::init_tensor(ntt_input, -10.f, 10.f);

    // ntt
    ntt::transpose<ntt::fixed_shape<perm_n, perm_c, perm_h, perm_w>>(
        ntt_input, ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    int64_t perms[] = {perm_n, perm_c, perm_h, perm_w, ndims};
    auto ort_output = ortki_Transpose(ort_input, perms, std::size(perms));

    // compare
    alignas(32) tensor_type2 ntt_output2(shape2);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(Transpose4DRankedShapePacked, NCHW) {
    transpose_4D_ranked_shape_packed<0, 1, 2, 3>();
}

TEST(Transpose4DRankedShapePacked, NCWH) {
    transpose_4D_ranked_shape_packed<0, 1, 3, 2>();
}

TEST(Transpose4DRankedShapePacked, NHCW) {
    transpose_4D_ranked_shape_packed<0, 2, 1, 3>();
}

TEST(Transpose4DRankedShapePacked, NHWC) {
    transpose_4D_ranked_shape_packed<0, 2, 3, 1>();
}

TEST(Transpose4DRankedShapePacked, NWCH) {
    transpose_4D_ranked_shape_packed<0, 3, 1, 2>();
}

TEST(Transpose4DRankedShapePacked, NWHC) {
    transpose_4D_ranked_shape_packed<0, 3, 2, 1>();
}

TEST(Transpose4DRankedShapePacked, CNHW) {
    transpose_4D_ranked_shape_packed<1, 0, 2, 3>();
}

TEST(Transpose4DRankedShapePacked, CNWH) {
    transpose_4D_ranked_shape_packed<1, 0, 3, 2>();
}

TEST(Transpose4DRankedShapePacked, CHNW) {
    transpose_4D_ranked_shape_packed<1, 2, 0, 3>();
}

TEST(Transpose4DRankedShapePacked, CHWN) {
    transpose_4D_ranked_shape_packed<1, 2, 3, 0>();
}

TEST(Transpose4DRankedShapePacked, CWNH) {
    transpose_4D_ranked_shape_packed<1, 3, 0, 2>();
}

TEST(Transpose4DRankedShapePacked, CWHN) {
    transpose_4D_ranked_shape_packed<1, 3, 2, 0>();
}

TEST(Transpose4DRankedShapePacked, HNCW) {
    transpose_4D_ranked_shape_packed<2, 0, 1, 3>();
}

TEST(Transpose4DRankedShapePacked, HNWC) {
    transpose_4D_ranked_shape_packed<2, 0, 3, 1>();
}

TEST(Transpose4DRankedShapePacked, HCNW) {
    transpose_4D_ranked_shape_packed<2, 1, 0, 3>();
}

TEST(Transpose4DRankedShapePacked, HCWN) {
    transpose_4D_ranked_shape_packed<2, 1, 3, 0>();
}

TEST(Transpose4DRankedShapePacked, HWNC) {
    transpose_4D_ranked_shape_packed<2, 3, 0, 1>();
}

TEST(Transpose4DRankedShapePacked, HWCN) {
    transpose_4D_ranked_shape_packed<2, 3, 1, 0>();
}

TEST(Transpose4DRankedShapePacked, WNCH) {
    transpose_4D_ranked_shape_packed<3, 0, 1, 2>();
}

TEST(Transpose4DRankedShapePacked, WNHC) {
    transpose_4D_ranked_shape_packed<3, 0, 2, 1>();
}

TEST(Transpose4DRankedShapePacked, WCNH) {
    transpose_4D_ranked_shape_packed<3, 1, 0, 2>();
}

TEST(Transpose4DRankedShapePacked, WCHN) {
    transpose_4D_ranked_shape_packed<3, 1, 2, 0>();
}

TEST(Transpose4DRankedShapePacked, WHNC) {
    transpose_4D_ranked_shape_packed<3, 2, 0, 1>();
}

TEST(Transpose4DRankedShapePacked, WHCN) {
    transpose_4D_ranked_shape_packed<3, 2, 1, 0>();
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
