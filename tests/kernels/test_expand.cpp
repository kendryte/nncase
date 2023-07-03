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

class ExpandTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, input_shape] = GetParam();

        input = hrt::create(typecode, input_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(Expand, ExpandTest,
                         testing::Combine(testing::Values(dt_float32, dt_int32,
                                                          dt_int64, dt_uint8,
                                                          dt_int8, dt_int16),
                                          testing::Values(dims_t{3, 1})));

TEST_P(ExpandTest, expand) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    int64_t new_shape[] = {1};
    auto new_shape_ptr = hrt::create(nncase::dt_int64, {1},
                                     {reinterpret_cast<gsl::byte *>(new_shape),
                                      sizeof(new_shape)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");
    auto new_shape_ort = runtime_tensor_2_ort_tensor(new_shape_ptr);
    auto output_ort = ortki_Expand(input_ort, new_shape_ort);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::expand(input.impl(), new_shape_ptr.impl())
                      .expect("expand failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));

    // expected
    int64_t new_shape1[] = {1, 1};
    auto new_shape_ptr1 =
        hrt::create(
            nncase::dt_int64, {2},
            {reinterpret_cast<gsl::byte *>(new_shape1), sizeof(new_shape1)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto new_shape_ort1 = runtime_tensor_2_ort_tensor(new_shape_ptr1);
    auto output_ort1 = ortki_Expand(input_ort, new_shape_ort1);
    size_t size1 = 0;
    void *ptr_ort1 = tensor_buffer(output_ort1, &size1);
    dims_t shape1(tensor_rank(output_ort1));
    tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
    auto expected1 =
        hrt::create(input.datatype(), shape1,
                    {reinterpret_cast<gsl::byte *>(ptr_ort1), size1}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output1 = kernels::stackvm::expand(input.impl(), new_shape_ptr1.impl())
                       .expect("expand failed");
    runtime_tensor actual1(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected1, actual1));

    // expected
    int64_t new_shape2[] = {3, 4};
    auto new_shape_ptr2 =
        hrt::create(
            nncase::dt_int64, {2},
            {reinterpret_cast<gsl::byte *>(new_shape2), sizeof(new_shape2)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto new_shape_ort2 = runtime_tensor_2_ort_tensor(new_shape_ptr2);
    auto output_ort2 = ortki_Expand(input_ort, new_shape_ort2);
    size_t size2 = 0;
    void *ptr_ort2 = tensor_buffer(output_ort2, &size2);
    dims_t shape2(tensor_rank(output_ort2));
    tensor_shape(output_ort2, reinterpret_cast<int64_t *>(shape2.data()));
    auto expected2 =
        hrt::create(input.datatype(), shape2,
                    {reinterpret_cast<gsl::byte *>(ptr_ort2), size2}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output2 = kernels::stackvm::expand(input.impl(), new_shape_ptr2.impl())
                       .expect("expand failed");
    runtime_tensor actual2(output2.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected2, actual2));

    // expected
    int64_t new_shape3[] = {2, 1, 6};
    auto new_shape_ptr3 =
        hrt::create(
            nncase::dt_int64, {3},
            {reinterpret_cast<gsl::byte *>(new_shape3), sizeof(new_shape3)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto new_shape_ort3 = runtime_tensor_2_ort_tensor(new_shape_ptr3);
    auto output_ort3 = ortki_Expand(input_ort, new_shape_ort3);
    size_t size3 = 0;
    void *ptr_ort3 = tensor_buffer(output_ort3, &size3);
    dims_t shape3(tensor_rank(output_ort3));
    tensor_shape(output_ort3, reinterpret_cast<int64_t *>(shape3.data()));
    auto expected3 =
        hrt::create(input.datatype(), shape3,
                    {reinterpret_cast<gsl::byte *>(ptr_ort3), size3}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output3 = kernels::stackvm::expand(input.impl(), new_shape_ptr3.impl())
                       .expect("expand failed");
    runtime_tensor actual3(output3.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected3, actual3) ||
                is_similarity_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}