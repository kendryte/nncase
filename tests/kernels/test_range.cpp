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

#define TEST_CASE_NAME "test_range"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class RangeTest : public KernelTest,
                  public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto shape = GetShapeArray("lhs_shape");
        auto begin_value = GetFloatNumber("begin");
        auto end_value = GetFloatNumber("end");
        auto step_value = GetFloatNumber("step");
        auto typecode = GetDataType("lhs_type");

        float_t begin_array[] = {begin_value};
        begin = hrt::create(typecode, shape,
                            {reinterpret_cast<gsl::byte *>(begin_array),
                             sizeof(begin_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        float_t end_array[] = {end_value};
        end = hrt::create(
                  typecode, shape,
                  {reinterpret_cast<gsl::byte *>(end_array), sizeof(end_array)},
                  true, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        float_t step_array[] = {step_value};
        step = hrt::create(typecode, shape,
                           {reinterpret_cast<gsl::byte *>(step_array),
                            sizeof(step_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor begin;
    runtime_tensor end;
    runtime_tensor step;
};

INSTANTIATE_TEST_SUITE_P(Range, RangeTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

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
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(begin, j)
    FOR_LOOP(end, k)
    FOR_LOOP(step, l)
    FOR_LOOP(lhs_type, m)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(begin, j)
    SPLIT_ELEMENT(end, k)
    SPLIT_ELEMENT(step, l)
    SPLIT_ELEMENT(lhs_type, m)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}