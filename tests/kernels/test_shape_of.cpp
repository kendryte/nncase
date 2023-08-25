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

#define TEST_CASE_NAME "test_shape_of"

class ShapeOfTest : public KernelTest,
                    public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {

        READY_SUBCASE()

        auto typecode = GetDataType("lhs_type");
        auto shape = GetShapeArray("i_shape");

        lhs = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);

        // expected
        size_t shape_size = shape.size();
        int64_t *shape_array = (int64_t *)malloc(shape_size * sizeof(int64_t));
        std::copy(shape.begin(), shape.end(), shape_array);
        expected = hrt::create(nncase::dt_int64, {shape_size},
                               {reinterpret_cast<gsl::byte *>(shape_array),
                                shape_size * sizeof(int64_t)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor lhs;
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(ShapeOf, ShapeOfTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ShapeOfTest, ShapeOf) {

    // actual
    auto output =
        kernels::stackvm::shape_of(lhs.impl()).expect("ShapeOf failed");
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
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(input_shape, k)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(input_shape, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}