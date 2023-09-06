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

#define TEST_CASE_NAME "test_gather_elements"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class GatherElementsTest : public KernelTest,
                           public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto shape = GetShapeArray("lhs_shape");
        auto indices_shape = GetShapeArray("indices_shape");
        auto indices_value = GetDataArray("indices_value");
        auto value = GetNumber("axis");
        auto typecode = GetDataType("lhs_type");

        input = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        size_t indices_value_size = indices_value.size();
        auto *indices_array =
            (int64_t *)malloc(indices_value_size * sizeof(int64_t));
        std::copy(indices_value.begin(), indices_value.end(), indices_array);
        indices = hrt::create(dt_int64, indices_shape,
                              {reinterpret_cast<gsl::byte *>(indices_array),
                               indices_value_size * sizeof(int64_t)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        batchDims_value = value;

        int64_t batchDims_array[1] = {batchDims_value};
        batchDims = hrt::create(dt_int64, dims_t{1},
                                {reinterpret_cast<gsl::byte *>(batchDims_array),
                                 sizeof(batchDims_array)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor batchDims;
    int64_t batchDims_value;
};

INSTANTIATE_TEST_SUITE_P(gather_elements, GatherElementsTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(GatherElementsTest, gather_elements) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);

    // expected
    auto output_ort =
        ortki_GatherElements(input_ort, indices_ort, batchDims_value);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::gather_elements(
                      input.impl(), batchDims.impl(), indices.impl())
                      .expect("gather failed");
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
    FOR_LOOP(indices_shape, l)
    FOR_LOOP(indices_value, h)
    FOR_LOOP(axis, j)
    FOR_LOOP(lhs_type, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(indices_shape, l)
    SPLIT_ELEMENT(indices_value, h)
    SPLIT_ELEMENT(axis, j)
    SPLIT_ELEMENT(lhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}