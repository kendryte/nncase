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

TEST(where, Pack) {
    constexpr size_t M = 2;
    constexpr size_t N = 2;
    // constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);
    float min_input = -100.0f;
    float max_input = 100.0f;

    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<M, N>>;
    using tensor_type2 = ntt::tensor<bool, ntt::fixed_shape<M, N>>;
    // init
    tensor_type2 condition;
    condition(0, 0) = true;
    condition(0, 1) = false;
    condition(1, 0) = false;
    condition(1, 1) = true;
    tensor_type1 ntt_input1;
    NttTest::init_tensor(ntt_input1, min_input, max_input);
    tensor_type1 ntt_input2;
    NttTest::init_tensor(ntt_input2, min_input, max_input);
    tensor_type1 ntt_output1;
    // ntt
    ntt::where<ntt::ops::where>(condition, ntt_input1, ntt_input2, ntt_output1);

    // ort
    auto ort_condition = NttTest::ntt2ort(condition);
    auto ort_input1 = NttTest::ntt2ort(ntt_input1);
    auto ort_input2 = NttTest::ntt2ort(ntt_input2);
    auto ort_output = ortki_Where(ort_condition, ort_input1, ort_input2);

    // compare
    alignas(2) tensor_type1 ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
