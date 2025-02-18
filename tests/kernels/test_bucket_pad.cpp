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

#define TEST_CASE_NAME "test_bucket_pad"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class BucketPadTest : public KernelTest,
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

        float value_array[] = {0};
        value = hrt::create(dt_float32, {1},
                            {reinterpret_cast<std::byte *>(value_array),
                             sizeof(value_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor value;
};

INSTANTIATE_TEST_SUITE_P(BucketPad, BucketPadTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(BucketPadTest, BucketPad) {

    // expected
    size_t size = 0;
    int64_t pad_ptr[] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto pad =
        hrt::create(dt_int64, {8},
                    {reinterpret_cast<std::byte *>(pad_ptr), sizeof(pad_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    int64_t axis_ptr[] = {0, 1, 2, 3};
    auto axis =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<std::byte *>(axis_ptr), sizeof(axis_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto l_ort = runtime_tensor_2_ort_tensor(input);
    auto pad_ort = runtime_tensor_2_ort_tensor(pad);
    auto value_ort = runtime_tensor_2_ort_tensor(value);
    auto axis_ort = runtime_tensor_2_ort_tensor(axis);
    auto output_ort =
        ortki_Pad(l_ort, pad_ort, value_ort, axis_ort, "constant");
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int64_t new_shape_array[] = {1, 3, 16, 16};
    auto new_shape =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<std::byte *>(new_shape_array),
                     sizeof(new_shape_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::bucket_pad(input.impl(), new_shape.impl())
                      .expect("pad failed");
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
    FOR_LOOP(lhs_shape, j)
    FOR_LOOP(lhs_type, i)
    SPLIT_ELEMENT(lhs_shape, j)
    SPLIT_ELEMENT(lhs_type, i)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}