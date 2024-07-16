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

TEST(MatmulTestFloat, NoPack) {
    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::matmul<false>(*ntt_lhs, *ntt_rhs, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_rhs;
    ntt::pack<1>(*ntt_lhs, p_ntt_lhs);
    ntt::pack<0>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, *ntt_output1,
                       ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{});

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_M) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_lhs;
    ntt::pack<0>(*ntt_lhs, p_ntt_lhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, *ntt_rhs, tmp, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{});
    unpack<0>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_rhs;
    ntt::pack<1>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        tmp;
    ntt::matmul<false>(*ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    unpack<1>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_rhs;
    ntt::pack<0>(*ntt_lhs, p_ntt_lhs);
    ntt::pack<1>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{});
    unpack<0, 1>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_lhs;
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        p_ntt_rhs;
    ntt::pack<0, 1>(*ntt_lhs, p_ntt_lhs);
    ntt::pack<0>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32 / P, 32>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                       ntt::fixed_shape<0>{});
    unpack<0>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_rhs;
    ntt::pack<1>(*ntt_lhs, p_ntt_lhs);
    ntt::pack<0, 1>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32) ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<32, 32 / P>>
        tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{});
    unpack<1>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(MatmulTestFloat, Pack_M_K_N) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 32>>;
    std::unique_ptr<tensor_type> ntt_lhs(new tensor_type);
    std::unique_ptr<tensor_type> ntt_rhs(new tensor_type);
    NttTest::init_tensor(*ntt_lhs, -2.f, 2.f);
    NttTest::init_tensor(*ntt_rhs, -2.f, 2.f);

    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_lhs;
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            p_ntt_rhs;
    ntt::pack<0, 1>(*ntt_lhs, p_ntt_lhs);
    ntt::pack<0, 1>(*ntt_rhs, p_ntt_rhs);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    alignas(32)
        ntt::tensor<ntt::vector<float, P, P>, ntt::fixed_shape<32 / P, 32 / P>>
            tmp;
    ntt::matmul<false>(p_ntt_lhs, p_ntt_rhs, tmp, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                       ntt::fixed_shape<0>{});
    unpack<0, 1>(tmp, *ntt_output1);

    // ort
    auto ort_lhs = NttTest::ntt2ort(*ntt_lhs);
    auto ort_rhs = NttTest::ntt2ort(*ntt_rhs);
    auto ort_output = ortki_MatMul(ort_lhs, ort_rhs);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
