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

class BroadCastTest : public KernelTest,
                      public ::testing::TestWithParam<
                          std::tuple<nncase::typecode_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, r_shape] = GetParam();

        float input_ptr[] = {3, 2, 1};
        input = hrt::create(typecode, l_shape,
                            {reinterpret_cast<gsl::byte *>(input_ptr),
                             sizeof(input_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        float output_ptr[] = {3, 2, 1, 3, 2, 1, 3, 2, 1};
        expected = hrt::create(typecode, r_shape,
                               {reinterpret_cast<gsl::byte *>(output_ptr),
                                sizeof(output_ptr)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(BroadCast, BroadCastTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{3}),
                                          testing::Values(dims_t{1, 3, 3})));

TEST_P(BroadCastTest, BroadCast) {
    //     actual
    int64_t a_ptr[] = {1, 3, 3};
    auto a = hrt::create(nncase::dt_int64, {3},
                         {reinterpret_cast<gsl::byte *>(a_ptr), sizeof(a_ptr)},
                         true, host_runtime_tensor::pool_cpu_only)
                 .expect("create tensor failed");
    auto output = kernels::stackvm::broadcast(input.impl(), a.impl())
                      .expect("broadcast failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        print_runtime_tensor(actual);
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}