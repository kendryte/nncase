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
#include "nncase/runtime/datatypes.h"
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

#define TEST_CASE_NAME "test_rank"

class RankTest : public KernelTest,
                 public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode_input = GetDataType("lhs_type");

        input = hrt::create(typecode_input, l_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        int64_t expected_array[] = {(int64_t)l_shape.size()};
        expected = hrt::create(dt_int64, {1},
                               {reinterpret_cast<gsl::byte *>(expected_array),
                                sizeof(expected_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(rank, RankTest,
                         testing::Combine(testing::Range(0, 100)));

TEST_P(RankTest, rank) {
    // actual
    int64_t shape_array[] = {1};
    auto shape = hrt::create(dt_int64, {1},
                             {reinterpret_cast<gsl::byte *>(shape_array),
                              sizeof(shape_array)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    auto output =
        kernels::stackvm::reshape(
            kernels::stackvm::rank(input.impl()).expect("rank failed"),
            shape.impl())
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
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(lhs_type, m)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, m)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}