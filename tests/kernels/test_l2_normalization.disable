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

class L2NormalizationTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        float lhs_ptr[] = {0.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
        size_t size = 0;
        lhs = hrt::create(lhs.datatype(), l_shape,
                          {reinterpret_cast<gsl::byte *>(lhs_ptr), size}, true,
                          host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        float expect_ptr[] = {0.0f, 0.4f, 0.6f, 0.4f, 0.4f, 0.4f};
        expect = hrt::create(expect.datatype(), l_shape,
                             {reinterpret_cast<gsl::byte *>(expect_ptr), size},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
    runtime_tensor expect;
};

INSTANTIATE_TEST_SUITE_P(L2Normalization, L2NormalizationTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{6})));

TEST_P(L2NormalizationTest, L2Normalization) {

    // expected
    auto expected = expect;

    // actual
    auto output = kernels::stackvm::l2_normalization(lhs.impl())
                      .expect("l2_normalization failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}