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

class RangeTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape] = GetParam();

        float_t begin_array[] = {0.0f};
        begin = hrt::create(typecode, shape,
                            {reinterpret_cast<gsl::byte *>(begin_array),
                             sizeof(begin_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        float_t end_array[] = {100.0f};
        end = hrt::create(
                  typecode, shape,
                  {reinterpret_cast<gsl::byte *>(end_array), sizeof(end_array)},
                  true, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        float_t step_array[] = {100.0f};
        step = hrt::create(typecode, shape,
                           {reinterpret_cast<gsl::byte *>(step_array),
                            sizeof(step_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor begin;
    runtime_tensor end;
    runtime_tensor step;
};

INSTANTIATE_TEST_SUITE_P(Range, RangeTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1})));

TEST_P(RangeTest, Range) {
    auto begin_ort = runtime_tensor_2_ort_tensor(begin);
    auto end_ort = runtime_tensor_2_ort_tensor(end);
    auto step_ort = runtime_tensor_2_ort_tensor(step);

    // expected
    auto output_ort = ortki_Range(begin_ort, end_ort, step_ort);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(begin.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::range(begin.impl(), end.impl(), step.impl())
                      .expect("range failed");
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