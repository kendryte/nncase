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

TEST(GatherTestFloat, no_pack_dynamic_shape_int32) {

    int32_t index_array[] = {1, 0, 2};
    auto tb = ntt::make_tensor_view(std::span<int32_t, 3>(index_array, 3),
                                    ntt::fixed_shape_v<1, 3>);

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 5, 8, 3>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);

    auto shape = ntt::make_shape(4, 5, 8, 3);
    auto ta_dynamic =
        ntt::make_tensor_view(std::span<float, std::dynamic_extent>(
                                  ta.elements().data(), shape.length()),
                              shape);
    ntt::gather(ta_dynamic, tb, tc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, no_pack_dynamic_shape_int64) {

    int64_t index_array[] = {1, 0, 2};
    auto tb = ntt::make_tensor_view(std::span<int64_t, 3>(index_array, 3),
                                    ntt::fixed_shape_v<1, 3>);

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 5, 8, 3>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);

    auto shape = ntt::make_shape(4, 5, 8, 3);
    auto ta_dynamic =
        ntt::make_tensor_view(std::span<float, std::dynamic_extent>(
                                  ta.elements().data(), shape.length()),
                              shape);
    ntt::gather(ta_dynamic, tb, tc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, no_pack_index_int32) {

    int32_t index_array[] = {1, 0, 2};
    auto tb = ntt::make_tensor_view(std::span<int32_t, 3>(index_array, 3),
                                    ntt::fixed_shape_v<1, 3>);

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 5, 8, 3>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);

    ntt::gather(ta, tb, tc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, no_pack_index_int64) {

    int64_t index_array[] = {1, 0, 2};
    auto tb = ntt::make_tensor_view(std::span<int64_t, 3>(index_array, 3),
                                    ntt::fixed_shape_v<1, 3>);

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 5, 8, 3>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 1, 3, 8, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);

    ntt::gather(ta, tb, tc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(ta);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // // compare
    NttTest::ort2ntt(ort_output, td);
    EXPECT_TRUE(NttTest::compare_tensor(tc, td));
}

TEST(GatherTestFloat, pack1d_dim0_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 1;

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, M / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::gather(pa, tb, pc, 0_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 0);

    // compare
    auto pd = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim0_no_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 32;
    constexpr size_t N = 128;
    constexpr size_t Period = 2;

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, M / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::gather(pa, tb, pc, 0_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 0);

    // compare
    auto pd = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<1, M / Period, N / P>);

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim1_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 4;
    constexpr size_t N = 512;
    constexpr size_t Period = 1;

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, N / P / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::gather(pa, tb, pc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    auto pd = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

TEST(GatherTestFloat, pack1d_dim1_no_contiguous) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    constexpr size_t M = 4;
    constexpr size_t N = 512;
    constexpr size_t Period = 2;

    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, N / P / Period>);
    auto pa =
        ntt::make_tensor<ntt::vector<float, P>>(ntt::fixed_shape_v<M, N / P>);
    auto pc = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0);
    std::ranges::for_each(tb.elements(), [](int64_t &x) { x *= Period; });
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::gather(pa, tb, pc, 1_dim);

    // ort
    auto ort_input = NttTest::ntt2ort(pa);
    auto ort_tb = NttTest::ntt2ort(tb);
    auto ort_output = ortki_Gather(ort_input, ort_tb, 1);

    // compare
    auto pd = ntt::make_tensor<ntt::vector<float, P>>(
        ntt::fixed_shape_v<M, 1, N / P / Period>);

    NttTest::ort2ntt(ort_output, pd);
    EXPECT_TRUE(NttTest::compare_tensor(pc, pd));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}