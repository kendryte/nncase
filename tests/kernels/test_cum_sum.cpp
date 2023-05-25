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
#include <c_api.h>
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <operators.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class CumSumTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
};

INSTANTIATE_TEST_SUITE_P(cum_sum, CumSumTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(CumSumTest, cum_sum) {
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);

    // expected
    float axis[] = {0};
    auto axis_ptr =
        hrt::create(nncase::dt_float32, {2},
                    {reinterpret_cast<gsl::byte *>(axis), sizeof(float)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto axis_ort = runtime_tensor_2_ort_tensor(axis_ptr);
    auto output_ort = ortki_CumSum(l_ort, axis_ort, 0, 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float exclusive[] = {0};
    auto exclusive_ptr =
        hrt::create(nncase::dt_float32, {2},
                    {reinterpret_cast<gsl::byte *>(exclusive), sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    float reverse[] = {0};
    auto reverse_ptr =
        hrt::create(nncase::dt_float32, {2},
                    {reinterpret_cast<gsl::byte *>(reverse), sizeof(float)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output =
        kernels::stackvm::cum_sum(lhs.impl(), axis_ptr.impl(),
                                  exclusive_ptr.impl(), reverse_ptr.impl())
            .expect("cum_sum failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}