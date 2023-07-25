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

class ClampTest
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
    clamp, ClampTest,
    testing::Combine(testing::Values(dt_float32, dt_int32, dt_int16, dt_int8,
                                     dt_uint8, dt_uint16, dt_uint32, dt_uint64,
                                     dt_int64, dt_float64, dt_boolean),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1},
                                     dims_t{1, 3}, dims_t{8, 8},
                                     dims_t{1, 3, 8}, dims_t{16, 16}, dims_t{},
                                     dims_t{16})));

TEST_P(ClampTest, clamp) {

    // expected
    float_t min1[] = {-1.0f};
    auto min_tensor1 =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(min1), sizeof(min1)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t max1[] = {1.0f};
    auto max_tensor1 =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(max1), sizeof(max1)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output1 = kernels::stackvm::clamp(input.impl(), min_tensor1.impl(),
                                           max_tensor1.impl())
                       .expect("clamp failed");
    runtime_tensor expected(output1.as<tensor>().expect("as tensor failed"));

    // actual
    float_t min[] = {-1.0f};
    auto min_tensor =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(min), sizeof(min)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float_t max[] = {1.0f};
    auto max_tensor =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(max), sizeof(max)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output = kernels::stackvm::clamp(input.impl(), min_tensor.impl(),
                                          max_tensor.impl())
                      .expect("clamp failed");
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