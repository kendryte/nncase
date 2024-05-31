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

#define TEST_CASE_NAME "test_scatter_nd"

class ScatterNDTest : public KernelTest,
                      public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode1 = GetDataType("lhs_type");
        auto typecode2 = GetDataType("rhs_type");
        auto input_shape = GetShapeArray("input_shape");
        auto indices_shape = GetShapeArray("indices_shape");
        auto updates_shape = GetShapeArray("updates_shape");

        input = hrt::create(typecode1, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        int64_t indices_array[] = {0, 0, 1, 1, 0, 1};
        indices = hrt::create(typecode2, indices_shape,
                              {reinterpret_cast<std::byte *>(indices_array),
                               sizeof(indices_array)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        updates = hrt::create(typecode1, updates_shape,
                              host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");
        init_tensor(updates);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor updates;
};

INSTANTIATE_TEST_SUITE_P(ScatterND, ScatterNDTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ScatterNDTest, ScatterND) {

    // expected
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);
    auto updates_ort = runtime_tensor_2_ort_tensor(updates);
    auto output_ort =
        ortki_ScatterND(input_ort, indices_ort, updates_ort, "none");
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::scatter_nd(input.impl(), indices.impl(),
                                               updates.impl())
                      .expect("scatter_nd failed");
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
    FOR_LOOP(rhs_type, j)
    FOR_LOOP(input_shape, k)
    FOR_LOOP(indices_shape, l)
    FOR_LOOP(updates_shape, m)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(rhs_type, j)
    SPLIT_ELEMENT(input_shape, k)
    SPLIT_ELEMENT(indices_shape, l)
    SPLIT_ELEMENT(updates_shape, m)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}