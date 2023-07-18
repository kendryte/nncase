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

class FlattenTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(
    flatten, FlattenTest,
    testing::Combine(
        testing::Values(dt_float32, dt_int8, dt_int32, dt_uint8, dt_int16),
        testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3, 48, 48})));

TEST_P(FlattenTest, flatten) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    auto output_ort = ortki_Flatten(l_ort, 1);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int32_t axis[] = {1};
    auto axis_ptr =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(axis), sizeof(axis)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::flatten(input.impl(), axis_ptr.impl())
                      .expect("flatten failed");
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

    //     expected
    auto output_ort1 = ortki_Flatten(l_ort, 2);
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
    int32_t axis1[] = {2};
    auto axis_ptr1 =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(axis1), sizeof(axis1)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output1 = kernels::stackvm::flatten(input.impl(), axis_ptr1.impl())
                       .expect("flatten failed");
    runtime_tensor actual1(output1.as<tensor>().expect("as tensor failed"));

    bool result1 = is_same_tensor(expected1, actual1) ||
                   cosine_similarity_tensor(expected1, actual1);

    if (!result1) {
        std::cout << "actual1 ";
        print_runtime_tensor(actual1);
        std::cout << "expected1 ";
        print_runtime_tensor(expected1);
    }

    // compare
    EXPECT_TRUE(result1);

    // expected
    auto output_ort2 = ortki_Flatten(l_ort, 3);
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
    int32_t axis2[] = {3};
    auto axis_ptr2 =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(axis2), sizeof(axis2)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output2 = kernels::stackvm::flatten(input.impl(), axis_ptr2.impl())
                       .expect("flatten failed");
    runtime_tensor actual2(output2.as<tensor>().expect("as tensor failed"));

    bool result2 = is_same_tensor(expected2, actual2) ||
                   cosine_similarity_tensor(expected2, actual2);

    if (!result2) {
        std::cout << "actual2 ";
        print_runtime_tensor(actual2);
        std::cout << "expected2 ";
        print_runtime_tensor(expected2);
    }

    // compare
    EXPECT_TRUE(result2);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}