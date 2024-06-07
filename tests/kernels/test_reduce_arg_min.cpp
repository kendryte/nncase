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

#define TEST_CASE_NAME "test_reduce_arg"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class ReduceArgMinTest : public KernelTest,
                         public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {

        READY_SUBCASE()

        auto typecode1 = GetDataType("lhs_type");
        auto typecode2 = GetDataType("rhs_type");
        auto l_shape = GetShapeArray("lhs_shape");
        auto r_shape = GetShapeArray("rhs_shape");
        auto value1 = GetNumber("axis_value");
        auto value2 = GetNumber("bool1_value");
        auto value3 = GetNumber("bool2_value");

        a = hrt::create(typecode1, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(a);
        axis_value = value1 > 0 ? value1 >= (int64_t)l_shape.size() ? 0 : value1
                     : -value1 > (int64_t)l_shape.size() ? 0
                                                         : value1;
        int64_t axis_array[] = {axis_value};
        axis = hrt::create(typecode2, r_shape,
                           {reinterpret_cast<std::byte *>(axis_array),
                            sizeof(axis_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
        keepDims_value = value2;
        int64_t keepDims_array[] = {keepDims_value};
        keepDims = hrt::create(typecode2, r_shape,
                               {reinterpret_cast<std::byte *>(keepDims_array),
                                sizeof(keepDims_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
        select_last_idx_value = value3;
        int64_t select_last_idx_array[] = {select_last_idx_value};
        select_last_idx =
            hrt::create(typecode2, r_shape,
                        {reinterpret_cast<std::byte *>(select_last_idx_array),
                         sizeof(select_last_idx_array)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor a;
    runtime_tensor axis;
    int64_t axis_value;
    runtime_tensor keepDims;
    int64_t keepDims_value;
    runtime_tensor select_last_idx;
    int64_t select_last_idx_value;
};

INSTANTIATE_TEST_SUITE_P(ReduceArgMin, ReduceArgMinTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ReduceArgMinTest, ReduceArgMin) {

    // expected
    size_t size = 0;
    auto output_ort = ortki_ArgMin(runtime_tensor_2_ort_tensor(a), axis_value,
                                   keepDims_value, select_last_idx_value);
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_int64, shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::reduce_arg(runtime::stackvm::reduce_arg_op_t::arg_min,
                                     dt_int64, a.impl(), axis.impl(),
                                     keepDims.impl(), select_last_idx.impl())
            .expect("reduce_arg_max failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "input ";
        print_runtime_tensor(a);
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
    FOR_LOOP(rhs_type, j)
    FOR_LOOP(lhs_shape, k)
    FOR_LOOP(rhs_shape, l)
    FOR_LOOP(bool1_value, m)
    FOR_LOOP(bool2_value, n)
    FOR_LOOP(axis_value, o)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(rhs_type, j)
    SPLIT_ELEMENT(lhs_shape, k)
    SPLIT_ELEMENT(rhs_shape, l)
    SPLIT_ELEMENT(bool1_value, m)
    SPLIT_ELEMENT(bool2_value, n)
    SPLIT_ELEMENT(axis_value, o)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}