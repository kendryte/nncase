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

class SizeOfTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(
    SizeOf, SizeOfTest,
    testing::Combine(testing::Values(dt_float32, dt_int8, dt_uint8, dt_boolean,
                                     dt_int32, dt_int16, dt_float64, dt_uint16),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3, 3, 3},
                                     dims_t{1, 3, 16}, dims_t{1},
                                     dims_t{1, 3})));

TEST_P(SizeOfTest, SizeOf) {

    // expected
    int64_t ptr_ort[] = {sizeof(input.shape())};
    auto expected =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(ptr_ort), sizeof(ptr_ort)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    int64_t shape_ort[] = {1};
    auto shape = hrt::create(dt_int64, {1},
                             {reinterpret_cast<gsl::byte *>(shape_ort),
                              sizeof(shape_ort)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    auto size_of_output =
        kernels::stackvm::size_of(input.impl()).expect("size_of failed");
    auto output = kernels::stackvm::reshape(size_of_output, shape.impl())
                      .expect("reshape failed");
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