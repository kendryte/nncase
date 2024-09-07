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

TEST(UnaryTestAcoshFloat, fixed_fixed) {
    // init
    using shape = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type = ntt::tensor<float, shape>;
    std::unique_ptr<tensor_type> ntt_input(new tensor_type);
    NttTest::init_tensor(*ntt_input, 1.f, 10.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::unary<ntt::ops::acosh>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Acosh(ort_input);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestAcoshFloat, fixed_ranked) {
    // init
    using shape1 = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type1 = ntt::tensor<float, shape1>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, 1.f, 10.f);

    // ntt
    auto shape2 = ntt::make_ranked_shape(1, 3, 16, 16);
    using tensor_type2 = ntt::tensor<float, ntt::ranked_shape<4>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2(shape2));
    ntt::unary<ntt::ops::acosh>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Acosh(ort_input);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2(shape2));
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestAcoshFloat, ranked_ranked) {
    // init
    using tensor_type = ntt::tensor<float, ntt::ranked_shape<4>>;
    auto shape = ntt::make_ranked_shape(1, 3, 16, 16);
    std::unique_ptr<tensor_type> ntt_input(new tensor_type(shape));
    NttTest::init_tensor(*ntt_input, 1.f, 10.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type(shape));
    ntt::unary<ntt::ops::acosh>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Acosh(ort_input);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type(shape));
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(UnaryTestAcoshFloat, ranked_fixed) {
    // init
    auto shape1 = ntt::make_ranked_shape(1, 3, 16, 16);
    using tensor_type1 = ntt::tensor<float, ntt::ranked_shape<4>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1(shape1));
    NttTest::init_tensor(*ntt_input, 1.f, 10.f);

    // ntt
    using shape2 = ntt::fixed_shape<1, 3, 16, 16>;
    using tensor_type2 = ntt::tensor<float, shape2>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::unary<ntt::ops::acosh>(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    auto ort_output = ortki_Acosh(ort_input);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

template <typename T, size_t vl> void test_vector() {
    ntt::vector<T, vl> ntt_input;
    NttTest::init_tensor(ntt_input, static_cast<T>(1), static_cast<T>(10));
    auto ntt_output1 = ntt::acosh(ntt_input);
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto ort_output = ortki_Acosh(ort_input);
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

TEST(UnaryTestAcosh, vector) { TEST_VECTOR(float) }

template <typename T, size_t vl> void test_vector_ulp(double ulp_threshold) {
    constexpr size_t size = ULP_SIZE;

    // init
    using tensor_type =
        ntt::tensor<ntt::vector<float, vl>, ntt::fixed_shape<size>>;
    std::unique_ptr<tensor_type> ntt_input(new tensor_type);
    NttTest::init_tensor(*ntt_input, 1.f, 10.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::unary<ntt::ops::acosh>(*ntt_input, *ntt_output1);

    // golden
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    nncase::ntt::apply(ntt_input->shape(), [&](auto index) {
        auto input_element = (*ntt_input)(index);
        auto &output_element = (*ntt_output2)(index);

        nncase::ntt::apply(input_element.shape(), [&](auto idx) {
            output_element(idx) = std::acosh(input_element(idx));
        });
    });

    // compare
    EXPECT_TRUE(
        NttTest::compare_ulp(*ntt_output1, *ntt_output2, ulp_threshold));
}

#define _TEST_VECTOR_ULP(T, lmul, ulp_threshold)                               \
    test_vector_ulp<T, (NTT_VLEN) / (sizeof(T) * 8) * lmul>(ulp_threshold);

#ifdef __riscv_vector
#define TEST_VECTOR_ULP(T, ulp_threshold)                                      \
    _TEST_VECTOR_ULP(T, 1, ulp_threshold)                                      \
    _TEST_VECTOR_ULP(T, 2, ulp_threshold)                                      \
    _TEST_VECTOR_ULP(T, 4, ulp_threshold)                                      \
    _TEST_VECTOR_ULP(T, 8, ulp_threshold)
#else
#define TEST_VECTOR_ULP(T, ulp_threshold) _TEST_VECTOR_ULP(T, 1, ulp_threshold)
#endif

// TEST(UnaryTestAcoshFloat, ulp_error) { TEST_VECTOR_ULP(float, 4.) }

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}