/* Copyright 2019-2021 Canaan Inc.
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
#include "kernel_test.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class SpaceToBatchTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        // expected
        float_t expected_array[] = {1, 3, 9,  11, 2, 4, 10, 12,
                                    5, 7, 13, 15, 6, 8, 14, 16};
        expected = hrt::create(typecode, l_shape,
                               {reinterpret_cast<gsl::byte *>(expected_array),
                                sizeof(expected_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(SpaceToBatch, SpaceToBatchTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{4, 2, 2, 1})));

TEST_P(SpaceToBatchTest, SpaceToBatch) {

    // actual
    float_t a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto input = hrt::create(dt_float32, {1, 4, 4, 1},
                             {reinterpret_cast<gsl::byte *>(a), sizeof(a)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    int64_t shape_array[] = {2, 2};
    auto shape = hrt::create(dt_int64, {2},
                             {reinterpret_cast<gsl::byte *>(shape_array),
                              sizeof(shape_array)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    int64_t crops_array[] = {0, 0, 0, 0};
    auto crops = hrt::create(dt_int64, {2, 2},
                             {reinterpret_cast<gsl::byte *>(crops_array),
                              sizeof(crops_array)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    auto output = kernels::stackvm::space_to_batch(input.impl(), shape.impl(),
                                                   crops.impl())
                      .expect("space_to_batch failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}