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

TEST(ScatterNDTestFloat, fixed_shape) {
    // Define tensor types
    using input_type = ntt::tensor<float, ntt::fixed_shape<5, 5, 5>>;
    using indices_type = ntt::tensor<int64_t, ntt::fixed_shape<2, 1>>;
    using updates_type = ntt::tensor<float, ntt::fixed_shape<2, 5, 5>>;
    using output_type = ntt::tensor<float, ntt::fixed_shape<5, 5, 5>>;

    // Initialize tensors
    input_type input;
    output_type ntt_output1;

    // Initialize input with zeros
    std::iota(input.elements().begin(), input.elements().end(), 0.f);

    std::unique_ptr<updates_type> updates(new updates_type);
    NttTest::init_tensor(*updates, -10.0f, 10.0f);

    // Setup indices and updates
    indices_type indices;
    indices(0, 0) = 1;
    indices(1, 0) = 2;

    // Perform scatterND
    ntt::scatter_nd(input, indices, *updates, ntt_output1);

    // Compare with ORT
    auto ort_input = NttTest::ntt2ort(input);
    auto ort_indices = NttTest::ntt2ort(indices);
    auto ort_updates = NttTest::ntt2ort(*updates);
    auto ort_output = ortki_ScatterND(ort_input, ort_indices, ort_updates, "none");    

    output_type ntt_output2;
    NttTest::ort2ntt(ort_output, ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
