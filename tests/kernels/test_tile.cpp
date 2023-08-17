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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
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

#define TEST_CASE_NAME "test_tile"

class TileTest : public KernelTest,
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

INSTANTIATE_TEST_SUITE_P(Tile, TileTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(TileTest, Tile) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    size_t size = 0;
    int64_t repeats_array[] = {1, 1, 2, 2};
    auto repeats = hrt::create(dt_int64, {4},
                               {reinterpret_cast<gsl::byte *>(repeats_array),
                                sizeof(repeats_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    auto output_ort =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats));
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::tile(input.impl(), repeats.impl())
                      .expect("tile failed");
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

    // expected
    int64_t repeats_array1[] = {1, 1, 1, 1};
    auto repeats1 = hrt::create(dt_int64, {4},
                                {reinterpret_cast<gsl::byte *>(repeats_array1),
                                 sizeof(repeats_array1)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto output_ort1 =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats1));
    void *ptr_ort1 = tensor_buffer(output_ort1, &size);
    dims_t shape1(tensor_rank(output_ort1));
    tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
    auto expected1 =
        hrt::create(input.datatype(), shape1,
                    {reinterpret_cast<gsl::byte *>(ptr_ort1), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output1 = kernels::stackvm::tile(input.impl(), repeats1.impl())
                       .expect("tile failed");
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
    int64_t repeats_array2[] = {1, 1, 3, 2};
    auto repeats2 = hrt::create(dt_int64, {4},
                                {reinterpret_cast<gsl::byte *>(repeats_array2),
                                 sizeof(repeats_array2)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto output_ort2 =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats2));
    void *ptr_ort2 = tensor_buffer(output_ort2, &size);
    dims_t shape2(tensor_rank(output_ort2));
    tensor_shape(output_ort2, reinterpret_cast<int64_t *>(shape2.data()));
    auto expected2 =
        hrt::create(input.datatype(), shape2,
                    {reinterpret_cast<gsl::byte *>(ptr_ort2), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output2 = kernels::stackvm::tile(input.impl(), repeats2.impl())
                       .expect("tile failed");
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

    // expected
    int64_t repeats_array3[] = {1, 1, 1, 2};
    auto repeats3 = hrt::create(dt_int64, {4},
                                {reinterpret_cast<gsl::byte *>(repeats_array3),
                                 sizeof(repeats_array3)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto output_ort3 =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats3));
    void *ptr_ort3 = tensor_buffer(output_ort3, &size);
    dims_t shape3(tensor_rank(output_ort3));
    tensor_shape(output_ort3, reinterpret_cast<int64_t *>(shape3.data()));
    auto expected3 =
        hrt::create(input.datatype(), shape3,
                    {reinterpret_cast<gsl::byte *>(ptr_ort3), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output3 = kernels::stackvm::tile(input.impl(), repeats3.impl())
                       .expect("tile failed");
    runtime_tensor actual3(output3.as<tensor>().expect("as tensor failed"));

    bool result3 = is_same_tensor(expected3, actual3) ||
                   cosine_similarity_tensor(expected3, actual3);

    if (!result3) {
        std::cout << "actual3 ";
        print_runtime_tensor(actual3);
        std::cout << "expected3 ";
        print_runtime_tensor(expected3);
    }

    // compare
    EXPECT_TRUE(result3);

    // expected
    int64_t repeats_array4[] = {1, 2, 3, 2};
    auto repeats4 = hrt::create(dt_int64, {4},
                                {reinterpret_cast<gsl::byte *>(repeats_array4),
                                 sizeof(repeats_array4)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto output_ort4 =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats4));
    void *ptr_ort4 = tensor_buffer(output_ort4, &size);
    dims_t shape4(tensor_rank(output_ort4));
    tensor_shape(output_ort4, reinterpret_cast<int64_t *>(shape4.data()));
    auto expected4 =
        hrt::create(input.datatype(), shape4,
                    {reinterpret_cast<gsl::byte *>(ptr_ort4), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output4 = kernels::stackvm::tile(input.impl(), repeats4.impl())
                       .expect("tile failed");
    runtime_tensor actual4(output4.as<tensor>().expect("as tensor failed"));

    bool result4 = is_same_tensor(expected4, actual4) ||
                   cosine_similarity_tensor(expected4, actual4);

    if (!result4) {
        std::cout << "actual4 ";
        print_runtime_tensor(actual4);
        std::cout << "expected4 ";
        print_runtime_tensor(expected4);
    }

    // compare
    EXPECT_TRUE(result4);

    // expected
    int64_t repeats_array5[] = {3, 2, 3, 2};
    auto repeats5 = hrt::create(dt_int64, {4},
                                {reinterpret_cast<gsl::byte *>(repeats_array5),
                                 sizeof(repeats_array5)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    auto output_ort5 =
        ortki_Tile(input_ort, runtime_tensor_2_ort_tensor(repeats5));
    void *ptr_ort5 = tensor_buffer(output_ort5, &size);
    dims_t shape5(tensor_rank(output_ort5));
    tensor_shape(output_ort5, reinterpret_cast<int64_t *>(shape5.data()));
    auto expected5 =
        hrt::create(input.datatype(), shape5,
                    {reinterpret_cast<gsl::byte *>(ptr_ort5), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    // actual
    auto output5 = kernels::stackvm::tile(input.impl(), repeats5.impl())
                       .expect("tile failed");
    runtime_tensor actual5(output5.as<tensor>().expect("as tensor failed"));

    bool result5 = is_same_tensor(expected5, actual5) ||
                   cosine_similarity_tensor(expected5, actual5);

    if (!result5) {
        std::cout << "actual5 ";
        print_runtime_tensor(actual5);
        std::cout << "expected5 ";
        print_runtime_tensor(expected5);
    }

    // compare
    EXPECT_TRUE(result5);
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(lhs_type, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}