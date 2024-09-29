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
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

template <typename T>
static ortki::OrtKITensor *ortki_SwishB(ortki::OrtKITensor *ort_input, T beta) {
    T one_data[1] = {static_cast<T>(1)};
    int64_t one_shape[1] = {1};
    auto ort_type = NttTest::primitive_type2ort_type<T>();
    auto one_tensor = make_tensor(reinterpret_cast<void *>(one_data), ort_type,
                                  one_shape, std::size(one_shape));

    T beta_data[1] = {beta};
    auto beta_tensor = make_tensor(reinterpret_cast<void *>(beta_data),
                                   ort_type, one_shape, std::size(one_shape));

    auto ort_neg = ortki_Neg(ort_input);
    auto ort_mul = ortki_Mul(ort_neg, beta_tensor);
    auto ort_exp = ortki_Exp(ort_mul);
    auto ort_add = ortki_Add(one_tensor, ort_exp);
    return ortki_Div(ort_input, ort_add);
}

template <typename T, size_t vl> void test_vector() {
    // init
    ntt::vector<T, vl> ntt_input;
    NttTest::init_tensor(ntt_input, static_cast<T>(-5), static_cast<T>(5));
    T beta = static_cast<T>(2);

    // ntt
    auto ntt_output1 = ntt::swishb(ntt_input, beta);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_SwishB<T>(ort_input, beta);
    ntt::vector<T, vl> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

#define _TEST_VECTOR(T, lmul)                                                  \
    test_vector<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>();

#ifdef __riscv_vector
#define TEST_VECTOR(T)                                                         \
    _TEST_VECTOR(T, 1)                                                         \
    _TEST_VECTOR(T, 2)                                                         \
    _TEST_VECTOR(T, 4)                                                         \
    _TEST_VECTOR(T, 8)
#else
#define TEST_VECTOR(T) _TEST_VECTOR(T, 1)
#endif

TEST(UnaryTestSwishB, vector) {
    TEST_VECTOR(float)
    TEST_VECTOR(double)
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}