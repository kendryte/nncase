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
#include <string_view>

using namespace nncase;
using namespace ortki;

template <typename T, size_t vl> void test_vector() {
    ntt::vector<T, vl> ntt_lhs, ntt_rhs;
    NttTest::init_tensor(ntt_lhs, static_cast<T>(-10), static_cast<T>(10));
    NttTest::init_tensor(ntt_rhs, static_cast<T>(-10), static_cast<T>(10));
    auto ntt_output1 = ntt::greater(ntt_lhs, ntt_rhs);
    auto ort_lhs = NttTest::ntt2ort(ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(ntt_rhs);
    auto ort_output = ortki_Greater(ort_lhs, ort_rhs);
    ntt::vector<bool, vl> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

#define _TEST_VECTOR(T, lmul)                                                  \
    test_vector<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>();

#define TEST_VECTOR(T)                                                         \
    _TEST_VECTOR(T, 1)                                                         \
    _TEST_VECTOR(T, 2)                                                         \
    _TEST_VECTOR(T, 4)                                                         \
    _TEST_VECTOR(T, 8)

TEST(CompareTestGreater, vector) {
    TEST_VECTOR(float)
    // TEST_VECTOR(int32_t)
    // TEST_VECTOR(int64_t)
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}