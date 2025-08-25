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

#define DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h, \
                                     expand_w)                                 \
    {                                                                          \
        /* init */                                                             \
        auto ntt_input =                                                       \
            ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);           \
        auto ntt_output1 = ntt::make_tensor<float>(                            \
            ntt::fixed_shape_v<expand_n, expand_c, expand_h, expand_w>);       \
                                                                               \
        std::iota(ntt_input.elements().begin(), ntt_input.elements().end(),    \
                  0.f);                                                        \
                                                                               \
        ntt::expand(ntt_input, ntt_output1);                                   \
                                                                               \
        auto ort_input = NttTest::ntt2ort(ntt_input);                          \
        int64_t data[] = {expand_n, expand_c, expand_h, expand_w};             \
        int64_t data_shape[] = {std::size(data)};                              \
        auto ort_type = NttTest::primitive_type2ort_type<int64_t>();           \
        auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,     \
                                 data_shape, std::size(data_shape));           \
        auto ort_output = ortki_Expand(ort_input, shape);                      \
                                                                               \
        auto ntt_output2 = ntt::make_tensor<float>(                            \
            ntt::fixed_shape_v<expand_n, expand_c, expand_h, expand_w>);       \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }

#define DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c,          \
                                      expand_h, expand_w)                      \
    {                                                                          \
        /* init */                                                             \
        auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W)); \
        auto ntt_output1 = ntt::make_tensor<float>(                            \
            ntt::make_shape(expand_n, expand_c, expand_h, expand_w));          \
                                                                               \
        std::iota(ntt_input.elements().begin(), ntt_input.elements().end(),    \
                  0.f);                                                        \
                                                                               \
        ntt::expand(ntt_input, ntt_output1);                                   \
                                                                               \
        auto ort_input = NttTest::ntt2ort(ntt_input);                          \
        int64_t data[] = {expand_n, expand_c, expand_h, expand_w};             \
        int64_t data_shape[] = {std::size(data)};                              \
        auto ort_type = NttTest::primitive_type2ort_type<int64_t>();           \
        auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,     \
                                 data_shape, std::size(data_shape));           \
        auto ort_output = ortki_Expand(ort_input, shape);                      \
                                                                               \
        auto ntt_output2 = ntt::make_tensor<float>(                            \
            ntt::make_shape(expand_n, expand_c, expand_h, expand_w));          \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }

TEST(ExpandFloat32, W) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, H) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, C) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, N) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, NC) {

    constexpr size_t N = 1;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, CH) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, HW) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, NH) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, CW) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(ExpandFloat32, NW) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                 expand_w);
}

TEST(RankedExpandFloat32, W) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, H) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, C) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, N) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, NC) {

    constexpr size_t N = 1;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, CH) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, HW) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, NH) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, CW) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandFloat32, NW) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_F32_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

#define DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c,          \
                                      expand_h, expand_w)                      \
    {                                                                          \
        /* init */                                                             \
        auto ntt_input =                                                       \
            ntt::make_tensor<float>(ntt::fixed_shape_v<N, C, H, W>);           \
        auto ntt_output1 = ntt::make_tensor<float>(                            \
            ntt::fixed_shape_v<expand_n, expand_c, expand_h, expand_w>);       \
                                                                               \
        std::iota(ntt_input.elements().begin(), ntt_input.elements().end(),    \
                  0.f);                                                        \
                                                                               \
        ntt::expand(ntt_input, ntt_output1);                                   \
                                                                               \
        auto ort_input = NttTest::ntt2ort(ntt_input);                          \
        int64_t data[] = {expand_n, expand_c, expand_h, expand_w};             \
        int64_t data_shape[] = {std::size(data)};                              \
        auto ort_type = NttTest::primitive_type2ort_type<int64_t>();           \
        auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,     \
                                 data_shape, std::size(data_shape));           \
        auto ort_output = ortki_Expand(ort_input, shape);                      \
                                                                               \
        auto ntt_output2 = ntt::make_tensor<float>(                            \
            ntt::fixed_shape_v<expand_n, expand_c, expand_h, expand_w>);       \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }

#define DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c,         \
                                       expand_h, expand_w)                     \
    {                                                                          \
        /* init */                                                             \
        auto ntt_input = ntt::make_tensor<float>(ntt::make_shape(N, C, H, W)); \
        auto ntt_output1 = ntt::make_tensor<float>(                            \
            ntt::make_shape(expand_n, expand_c, expand_h, expand_w));          \
                                                                               \
        std::iota(ntt_input.elements().begin(), ntt_input.elements().end(),    \
                  0.f);                                                        \
                                                                               \
        ntt::expand(ntt_input, ntt_output1);                                   \
                                                                               \
        auto ort_input = NttTest::ntt2ort(ntt_input);                          \
        int64_t data[] = {expand_n, expand_c, expand_h, expand_w};             \
        int64_t data_shape[] = {std::size(data)};                              \
        auto ort_type = NttTest::primitive_type2ort_type<int64_t>();           \
        auto shape = make_tensor(reinterpret_cast<void *>(data), ort_type,     \
                                 data_shape, std::size(data_shape));           \
        auto ort_output = ortki_Expand(ort_input, shape);                      \
                                                                               \
        auto ntt_output2 = ntt::make_tensor<float>(                            \
            ntt::make_shape(expand_n, expand_c, expand_h, expand_w));          \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }

TEST(ExpandHalf, W) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, H) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, C) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, N) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, NC) {

    constexpr size_t N = 1;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, CH) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, HW) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, NH) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, CW) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(ExpandHalf, NW) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_FIXED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                  expand_w);
}

TEST(RankedExpandHalf, W) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, H) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, C) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, N) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, NC) {

    constexpr size_t N = 1;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, CH) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, HW) {

    constexpr size_t N = 8;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, NH) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 1;
    constexpr size_t W = 8;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, CW) {

    constexpr size_t N = 8;
    constexpr size_t C = 1;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

TEST(RankedExpandHalf, NW) {

    constexpr size_t N = 1;
    constexpr size_t C = 8;
    constexpr size_t H = 8;
    constexpr size_t W = 1;

    constexpr size_t expand_n = 8;
    constexpr size_t expand_c = 8;
    constexpr size_t expand_h = 8;
    constexpr size_t expand_w = 8;

    DEFINE_RANKED_EXPAND_HALF_TEST(N, C, H, W, expand_n, expand_c, expand_h,
                                   expand_w);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
