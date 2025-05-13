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
#include "nncase/ntt/tensor.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>
#include <string_view>

using namespace nncase;
using namespace ortki;

#define _TEST_VECTOR(T, lmul)                                                  \
    test_vector<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>();

#define TEST_VECTOR(T)                                                         \
    _TEST_VECTOR(T, 1)                                                         \
    _TEST_VECTOR(T, 2)                                                         \
    _TEST_VECTOR(T, 4)                                                         \
    _TEST_VECTOR(T, 8)

#define DEFINE_NTT_BINARY_TEST(ntt_name, ort_name)                             \
    TEST(BinaryTest##ort_name##Float, fixed_fixed_fixed) {                     \
        /* init */                                                             \
        auto shape = fixed_shape_v<1, 3, 16, 16>;                              \
        auto ntt_lhs = make_unique_tensor<float>(shape);                       \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_fixed_fixed_broadcast_lhs_scalar) {                             \
        /* init */                                                             \
        auto shape1 = fixed_shape_v<1>;                                        \
        auto ntt_rhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = fixed_shape_v<1, 3, 16, 16>;                             \
        auto ntt_lhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape2);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape2);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_fixed_fixed_broadcast_rhs_scalar) {                             \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::fixed_shape_v<1>;                                   \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape1);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape1);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_fixed_fixed_broadcast_lhs_vector) {                             \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<16>;                                  \
        auto ntt_rhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_lhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape2);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape2);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_fixed_fixed_broadcast_rhs_vector) {                             \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::fixed_shape_v<16>;                                  \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape1);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape1);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_fixed_fixed_broadcast_multidirectional) {                       \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 1, 16>;                         \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::fixed_shape_v<3, 1, 16, 1>;                         \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::fixed_shape_v<3, 3, 16, 16>;                        \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float, fixed_ranked_ranked) {                   \
        /* init */                                                             \
        auto shape1 = fixed_shape_v<1, 3, 16, 16>;                             \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::make_shape(1, 3, 16, 16);                            \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_ranked_ranked_broadcast_lhs_scalar) {                           \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1>;                                   \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::make_shape(1, 3, 16, 16);                            \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_ranked_ranked_broadcast_rhs_scalar) {                           \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::make_shape(1);                                       \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_ranked_ranked_broadcast_lhs_vector) {                           \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<16>;                                  \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::make_shape(1, 3, 16, 16);                            \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_ranked_ranked_broadcast_rhs_vector) {                           \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::make_shape(16);                                      \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         fixed_ranked_ranked_broadcast_multidirectional) {                     \
        /* init */                                                             \
        auto shape1 = ntt::fixed_shape_v<1, 3, 1, 16>;                         \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(3, 1, 16, 1);                            \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(3, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float, ranked_fixed_ranked) {                   \
        /* init */                                                             \
        auto shape = ntt::make_shape(1, 3, 16, 16);                            \
        auto ntt_lhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape1 = ntt::fixed_shape_v<1, 3, 16, 16>;                        \
        auto ntt_rhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_fixed_ranked_broadcast_lhs_scalar) {                           \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1);                                      \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::fixed_shape_v<1, 3, 16, 16>;                         \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_fixed_ranked_broadcast_rhs_scalar) {                           \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::fixed_shape_v<1>;                                    \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_fixed_ranked_broadcast_lhs_vector) {                           \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(16);                                     \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::fixed_shape_v<1, 3, 16, 16>;                         \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_fixed_ranked_broadcast_rhs_vector) {                           \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape = ntt::fixed_shape_v<16>;                                   \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_fixed_ranked_broadcast_multidirectional) {                     \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 1, 16);                            \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::fixed_shape_v<3, 1, 16, 1>;                         \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(3, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float, ranked_ranked_ranked) {                  \
        /* init */                                                             \
        auto shape = ntt::make_shape(1, 3, 16, 16);                            \
        auto ntt_lhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto ntt_rhs = make_unique_tensor<float>(shape);                       \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape);                   \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape);                   \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_ranked_ranked_broadcast_lhs_scalar) {                          \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1);                                      \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape2);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape2);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_ranked_ranked_broadcast_rhs_scalar) {                          \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(1);                                      \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape1);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape1);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_ranked_ranked_broadcast_lhs_vector) {                          \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(16);                                     \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape2);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape2);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_ranked_ranked_broadcast_rhs_vector) {                          \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 16, 16);                           \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(16);                                     \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto ntt_output1 = make_unique_tensor<float>(shape1);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape1);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    TEST(BinaryTest##ort_name##Float,                                          \
         ranked_ranked_ranked_broadcast_multidirectional) {                    \
        /* init */                                                             \
        auto shape1 = ntt::make_shape(1, 3, 1, 16);                            \
        auto ntt_lhs = make_unique_tensor<float>(shape1);                      \
        NttTest::init_tensor(*ntt_lhs, -10.f, 10.f);                           \
                                                                               \
        auto shape2 = ntt::make_shape(3, 1, 16, 1);                            \
        auto ntt_rhs = make_unique_tensor<float>(shape2);                      \
        NttTest::init_tensor(*ntt_rhs, -10.f, 10.f);                           \
                                                                               \
        /* ntt */                                                              \
        auto shape3 = ntt::make_shape(3, 3, 16, 16);                           \
        auto ntt_output1 = make_unique_tensor<float>(shape3);                  \
        ntt::binary<ntt::ops::ntt_name>(*ntt_lhs, *ntt_rhs, *ntt_output1);     \
                                                                               \
        /* ort */                                                              \
        auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);                             \
        auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);                             \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
                                                                               \
        /* compare */                                                          \
        auto ntt_output2 = make_unique_tensor<float>(shape3);                  \
        NttTest::ort2ntt(ort_output, *ntt_output2);                            \
        EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));      \
    }                                                                          \
                                                                               \
    template <typename T, size_t vl> void test_vector() {                      \
        ntt::vector<T, vl> ntt_lhs, ntt_rhs;                                   \
        NttTest::init_tensor(ntt_lhs, static_cast<T>(-10),                     \
                             static_cast<T>(10));                              \
        NttTest::init_tensor(ntt_rhs, static_cast<T>(-10),                     \
                             static_cast<T>(10));                              \
        auto ntt_output1 = ntt::ntt_name(ntt_lhs, ntt_rhs);                    \
        auto ort_lhs = NttTest::ntt2ort(ntt_lhs);                              \
        auto ort_rhs = NttTest::ntt2ort(ntt_rhs);                              \
        auto ort_output = ortki_##ort_name(ort_lhs, ort_rhs);                  \
        ntt::vector<T, vl> ntt_output2;                                        \
        NttTest::ort2ntt(ort_output, ntt_output2);                             \
        EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));        \
    }                                                                          \
    TEST(UnaryTest##ort_name##, vector) {                                      \
        TEST_VECTOR(float)                                                     \
        TEST_VECTOR(int32_t)                                                   \
        TEST_VECTOR(int64_t)                                                   \
    }                                                                          \
                                                                               \
    int main(int argc, char *argv[]) {                                         \
        ::testing::InitGoogleTest(&argc, argv);                                \
        return RUN_ALL_TESTS();                                                \
    }
