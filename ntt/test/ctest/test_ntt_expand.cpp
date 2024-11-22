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

TEST(ExpandTestFloat, NoPack) {
    constexpr size_t M = 1024;
    constexpr size_t N = 1;
    constexpr size_t K = 2048;

    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);

    // init
    using input_tensor_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using output_tensor_type = ntt::tensor<float, ntt::fixed_shape<M, K>>;

    std::unique_ptr<input_tensor_type> ntt_input(new input_tensor_type);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<output_tensor_type> ntt_output1(new output_tensor_type);
    ntt::expand(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t target_shape[] = {M, K};
    int64_t shape_size = 2;
    int64_t shape[] = {shape_size};
    auto shape_tensor = make_tensor(reinterpret_cast<void*>(target_shape), DataType_INT64, shape, 1);
    auto ort_output = ortki_Expand(ort_input, shape_tensor);

    // compare
    std::unique_ptr<output_tensor_type> ntt_output2(new output_tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ExpandTestFloat, NoPack1) {
    constexpr size_t M = 1;
    constexpr size_t K = 2;

    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);

    // init
    using input_tensor_type = ntt::tensor<float, ntt::fixed_shape<M>>;
    using output_tensor_type = ntt::tensor<float, ntt::fixed_shape<M, K>>;
    std::unique_ptr<input_tensor_type> ntt_input(new input_tensor_type);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

   // ntt
    std::unique_ptr<output_tensor_type> ntt_output1(new output_tensor_type);
    ntt::expand(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t target_shape[] = {M, K};
    int64_t shape_size = 2;
    int64_t shape[] = {shape_size};
    auto shape_tensor = make_tensor(reinterpret_cast<void*>(target_shape), DataType_INT64, shape, 1);
    auto ort_output = ortki_Expand(ort_input, shape_tensor);

    // compare
    std::unique_ptr<output_tensor_type> ntt_output2(new output_tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ExpandTestFloat, Pack_M_K) {
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    // init
    using input_tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 1>>;
    std::unique_ptr<input_tensor_type> ntt_input(new input_tensor_type);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    constexpr size_t pack_dim = (31 + P) / P; 
    alignas(32) ntt::tensor<ntt::vector<float, 128>, ntt::fixed_shape<pack_dim, 1>> p_ntt_lhs;
    ntt::pack<0>(*ntt_input, p_ntt_lhs);

    // ntt
    using output_tensor_type = ntt::tensor<float, ntt::fixed_shape<32, 2>>;
    std::unique_ptr<output_tensor_type> ntt_output1(new output_tensor_type);
    ntt::expand(*ntt_input, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t target_shape[] = {32, 2};
    int64_t shape_size = 2;
    int64_t shape[] = {shape_size};
    auto shape_tensor = make_tensor(reinterpret_cast<void*>(target_shape), DataType_INT64, shape, 1);
    auto ort_output = ortki_Expand(ort_input, shape_tensor);
    std::unique_ptr<output_tensor_type> ntt_output2(new output_tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);

    // compare
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}