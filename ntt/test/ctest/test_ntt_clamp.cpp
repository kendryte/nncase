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

TEST(ClampTestFloat, NoPack) {
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    using tensor_type = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    std::unique_ptr<tensor_type> ntt_input(new tensor_type);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    std::unique_ptr<tensor_type> ntt_output1(new tensor_type);
    ntt::clamp(*ntt_input, *ntt_output1, min_clamp, max_clamp);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    std::unique_ptr<tensor_type> ntt_output2(new tensor_type);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ClampTestFloat, PackM) {
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M / P, N>>;
    std::unique_ptr<tensor_type2> pack_input(new tensor_type2);
    std::unique_ptr<tensor_type2> pack_output(new tensor_type2);
    ntt::pack<0>(*ntt_input, *pack_input);
    ntt::clamp(*pack_input, *pack_output, min_clamp, max_clamp);
    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    ntt::unpack<0>(*pack_output, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ClampTestFloat, PackN) {
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, min_input, max_input);

    // ntt
    using tensor_type2 =
        ntt::tensor<ntt::vector<float, P>, ntt::fixed_shape<M, N / P>>;
    std::unique_ptr<tensor_type2> pack_input(new tensor_type2);
    std::unique_ptr<tensor_type2> pack_output(new tensor_type2);
    ntt::pack<1>(*ntt_input, *pack_input);
    ntt::clamp(*pack_input, *pack_output, min_clamp, max_clamp);
    std::unique_ptr<tensor_type1> ntt_output1(new tensor_type1);
    ntt::unpack<1>(*pack_output, *ntt_output1);

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    std::unique_ptr<tensor_type1> ntt_output2(new tensor_type1);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
