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

#define TEST_CASE_NAME "test_unsqueeze"

class UnsqueezeTest : public KernelTest,
                      public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto axis_value = GetAxesArray("axis_array");
        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        axis_array = axis_value;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    axes_t axis_array;
};

INSTANTIATE_TEST_SUITE_P(Unsqueeze, UnsqueezeTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(UnsqueezeTest, Unsqueeze) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    if (axis_array.size() + input.shape().size() == 4) {

        // expected
        size_t axis_size = axis_array.size();
        int64_t *axis_array1 = (int64_t *)malloc(axis_size * sizeof(int64_t));
        std::copy(axis_array.begin(), axis_array.end(), axis_array1);
        auto axes = hrt::create(dt_int64, {axis_size},
                                {reinterpret_cast<gsl::byte *>(axis_array1),
                                 axis_size * sizeof(int64_t)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
        auto output_ort =
            ortki_Unsqueeze(l_ort, runtime_tensor_2_ort_tensor(axes));

        size_t size = 0;
        void *ptr_ort = tensor_buffer(output_ort, &size);
        dims_t shape(tensor_rank(output_ort));
        tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
        auto expected =
            hrt::create(input.datatype(), shape,
                        {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output = kernels::stackvm::unsqueeze(input.impl(), axes.impl())
                          .expect("unsqueeze failed");
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
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(axis_array, j)
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(lhs_type, k)
    SPLIT_ELEMENT(axis_array, j)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}