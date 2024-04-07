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

#define TEST_CASE_NAME "test_cum_sum"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class CumSumTest : public KernelTest,
                   public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(cum_sum, CumSumTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(CumSumTest, cum_sum) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    int64_t axis[] = {1};
    auto axis_ptr = hrt::create(nncase::dt_int64, {1},
                                {reinterpret_cast<std::byte *>(axis), 8}, true,
                                host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto axis_ort = runtime_tensor_2_ort_tensor(axis_ptr);
    auto output_ort = ortki_CumSum(l_ort, axis_ort, 0, 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    float exclusive[] = {0};
    auto exclusive_ptr = hrt::create(nncase::dt_float32, {1},
                                     {reinterpret_cast<std::byte *>(exclusive),
                                      sizeof(exclusive)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");
    float reverse[] = {0};
    auto reverse_ptr =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<std::byte *>(reverse), sizeof(reverse)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output =
        kernels::stackvm::cum_sum(input.impl(), axis_ptr.impl(),
                                  exclusive_ptr.impl(), reverse_ptr.impl())
            .expect("cum_sum failed");
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
    FOR_LOOP(lhs_type, j)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}