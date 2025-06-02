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
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(ClampTestFloat, NoPack) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    auto shape1 = ntt::fixed_shape_v<M, N>;
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::clamp(ntt_input, ntt_output1, min_clamp, max_clamp);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(ClampTestFloat, PackM) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    auto shape1 = ntt::fixed_shape_v<M, N>;
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto shape2 = ntt::fixed_shape_v<M / P, N>;
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<0>);
    ntt::clamp(pack_input, pack_output, min_clamp, max_clamp);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<0>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

TEST(ClampTestFloat, PackN) {
    constexpr dim_t M = 32;
    constexpr dim_t N = 32;
    constexpr dim_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = static_cast<float>(-10);
    float max_input = static_cast<float>(10);
    float min_clamp = static_cast<float>(-6);
    float max_clamp = static_cast<float>(6);

    // init
    auto shape1 = ntt::fixed_shape_v<M, N>;
    alignas(32) auto ntt_input = ntt::make_tensor<float>(shape1);
    NttTest::init_tensor(ntt_input, min_input, max_input);

    // ntt
    auto shape2 = ntt::fixed_shape_v<M, N / P>;
    alignas(32) auto pack_input =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    alignas(32) auto pack_output =
        ntt::make_tensor<ntt::vector<float, P>>(shape2);
    ntt::pack(ntt_input, pack_input, ntt::fixed_shape_v<1>);
    ntt::clamp(pack_input, pack_output, min_clamp, max_clamp);
    alignas(32) auto ntt_output1 = ntt::make_tensor<float>(shape1);
    ntt::unpack(pack_output, ntt_output1, ntt::fixed_shape_v<1>);

    // ort
    auto ort_input = NttTest::ntt2ort(ntt_input);
    float min_buf[] = {min_clamp};
    int64_t shape[] = {std::size(min_buf)};
    auto min = make_tensor(reinterpret_cast<void *>(min_buf), DataType_FLOAT,
                           shape, 1);
    float max_buf[] = {max_clamp};
    auto max = make_tensor(reinterpret_cast<void *>(max_buf), DataType_FLOAT,
                           shape, 1);
    auto ort_output = ortki_Clip(ort_input, min, max);

    // compare
    alignas(32) auto ntt_output2 = ntt::make_tensor<float>(shape1);
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
