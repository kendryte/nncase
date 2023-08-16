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

#define TEST_CASE_NAME "test_batch_to_space"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class BatchToSpaceTest : public KernelTest,
                         public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto input_shape = GetShapeArray("lhs_shape");
        auto expect_shape = GetShapeArray("rhs_shape");
        auto typecode = GetDataType("lhs_type");

        input = hrt::create(typecode, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        expect = hrt::create(typecode, expect_shape,
                             host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
        init_tensor(expect);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor expect;
};

INSTANTIATE_TEST_SUITE_P(BatchToSpace, BatchToSpaceTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(BatchToSpaceTest, BatchToSpace) {

    // expected
    float_t b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto b_ptr = b;
    auto expected = hrt::create(input.datatype(), expect.shape(),
                                {reinterpret_cast<gsl::byte *>(b_ptr), 64},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float_t a[] = {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16};
    auto input_tensor =
        hrt::create(input.datatype(), input.shape(),
                    {reinterpret_cast<gsl::byte *>(a), sizeof(a)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    int64_t crops[] = {0, 0, 0, 0};
    auto crops_tensor = hrt::create(dt_int64, {2, 2},
                                    {reinterpret_cast<gsl::byte *>(crops), 32},
                                    true, host_runtime_tensor::pool_cpu_only)
                            .expect("create tensor failed");
    int64_t shape[] = {2, 2};
    auto shape_tensor =
        hrt::create(dt_int64, {2},
                    {reinterpret_cast<gsl::byte *>(shape), sizeof(shape)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::batch_to_space(input_tensor.impl(),
                                                   shape_tensor.impl(),
                                                   crops_tensor.impl())
                      .expect("batch_to_space failed");
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
    FOR_LOOP(rhs_shape, j)
    FOR_LOOP(lhs_type, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(rhs_shape, j)
    SPLIT_ELEMENT(lhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}