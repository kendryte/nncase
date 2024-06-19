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

TEST(UnaryTestExpFloat, fixed_fixed) {
    // init
    using shape = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type = ntt::tensor<float, shape>;
    std::unique_ptr<tensor_type> ntt_input(new tensor_type);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::unary<ntt::ops::exp>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Exp(ort_input);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestExpFloat, fixed_ranked) {
    // init
    using shape1 = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type1 = ntt::tensor<float, shape1>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    auto shape2 = ntt::make_ranked_shape(1, 3, 16, 16);
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<4>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2(shape2));
    ntt::unary<ntt::ops::exp>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Exp(ort_input);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2(shape2));
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestExpFloat, ranked_ranked) {
    // init
    using tensor_type = ntt::tensor<float, ntt::ranked_shape<4>>;
    auto shape = ntt::make_ranked_shape(1, 3, 16, 16);
    std::unique_ptr<tensor_type> ntt_input(new tensor_type(shape));
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type(shape));
    ntt::unary<ntt::ops::exp>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Exp(ort_input);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type(shape));
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestExpFloat, ranked_fixed) {
    // init
    auto shape1 = ntt::make_ranked_shape(1, 3, 16, 16);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1(shape1));
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    using shape2 = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type2 = ntt::tensor<float, shape2>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::unary<ntt::ops::exp>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Exp(ort_input);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

template <typename T, size_t vl> void test_vector() {
    ntt::vector<T, vl> ntt_input;
    NttTest::init_tensor(ntt_input, static_cast<T>(-10), static_cast<T>(10));
    auto ntt_output1 = ntt::exp(ntt_input);
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Exp(ort_input);
    ntt::vector<T, vl> ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2, 0.999999f));
}

#define _TEST_VECTOR(T, lmul)                                                  \
    test_vector<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>();

#define TEST_VECTOR(T)                                                         \
    _TEST_VECTOR(T, 1)                                                         \
    _TEST_VECTOR(T, 2)                                                         \
    _TEST_VECTOR(T, 4)                                                         \
    _TEST_VECTOR(T, 8)

TEST(UnaryTestExp, vector) { TEST_VECTOR(float) }

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}