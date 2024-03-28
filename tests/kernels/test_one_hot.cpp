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

#define TEST_CASE_NAME "test_one_hot"

class OneHotTest : public KernelTest,
                   public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {

        READY_SUBCASE()
        auto l_shape = GetShapeArray("lhs_shape");
        auto values_shape = GetShapeArray("rhs_shape");
        auto axis_value = GetNumber("axis");
        auto value_typecode = GetDataType("lhs_type");
        auto index_typecode = GetDataType("rhs_type");

        int64_t a[] = {3, 2, 4, 0};
        indices = hrt::create(index_typecode, l_shape,
                              {reinterpret_cast<std::byte *>(a), sizeof(a)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        values = hrt::create(value_typecode, values_shape,
                             host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
        init_tensor(values);

        int32_t depth_ptr[] = {5};
        depth = hrt::create(dt_int32, {1},
                            {reinterpret_cast<std::byte *>(depth_ptr),
                             sizeof(depth_ptr)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        axis = axis_value;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor indices;
    runtime_tensor values;
    runtime_tensor depth;
    int32_t axis;
};

INSTANTIATE_TEST_SUITE_P(OneHot, OneHotTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(OneHotTest, OneHot) {
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);
    auto values_ort = runtime_tensor_2_ort_tensor(values);
    auto depth_ort = runtime_tensor_2_ort_tensor(depth);

    // expected
    auto output_ort = ortki_OneHot(indices_ort, depth_ort, values_ort, axis);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(values.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int axis_ptr[] = {axis};
    auto axis =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<std::byte *>(axis_ptr), sizeof(axis_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::one_hot(
                      runtime::stackvm::one_hot_mode_t::process_neg,
                      indices.impl(), depth.impl(), values.impl(), axis.impl())
                      .expect("one_hot failed");
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
    FOR_LOOP(axis, k)
    FOR_LOOP(lhs_type, l)
    FOR_LOOP(rhs_type, m)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(rhs_shape, j)
    SPLIT_ELEMENT(axis, k)
    SPLIT_ELEMENT(lhs_type, l)
    SPLIT_ELEMENT(rhs_type, m)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}