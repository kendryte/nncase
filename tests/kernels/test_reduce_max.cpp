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

class ReduceMaxTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<
          nncase::typecode_t, typecode_t, dims_t, dims_t, int64_t, axes_t>> {
  public:
    void SetUp() override {
        auto &&[typecode1, typecode2, l_shape, r_shape, value, axis_arry] =
            GetParam();

        a = hrt::create(typecode1, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(a);

        keepDims_value = value;
        int64_t keepDims_array[] = {keepDims_value};
        keepDims = hrt::create(typecode2, r_shape,
                               {reinterpret_cast<gsl::byte *>(keepDims_array),
                                sizeof(keepDims_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");

        int64_t init_value_array[] = {0};
        init_value =
            hrt::create(typecode2, r_shape,
                        {reinterpret_cast<gsl::byte *>(init_value_array),
                         sizeof(init_value_array)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        axis_arry1 = axis_arry;
    }

    void TearDown() override {}

  protected:
    runtime_tensor a;
    axes_t axis_arry1;
    int64_t keepDims_value;
    runtime_tensor keepDims;
    runtime_tensor init_value;
};

INSTANTIATE_TEST_SUITE_P(
    ReduceMax, ReduceMaxTest,
    testing::Combine(testing::Values(dt_float32), testing::Values(dt_int64),
                     testing::Values(dims_t{1, 3, 16, 16}),
                     testing::Values(dims_t{1}), testing::Values(0, 1),
                     testing::Values(/*axes_t{0},*/ axes_t{-1},
                                     axes_t{-2}, /*axes_t{-3}, axes_t{1},*/
                                     axes_t{2}, axes_t{3}, axes_t{2, 3},
                                     axes_t{-2, -1}, axes_t{1, 2, 3},
                                     axes_t{-1, -2, -3}, axes_t{0, 1, 2, 3},
                                     axes_t{-1, -2, -3, -4})));

TEST_P(ReduceMaxTest, ReduceMax) {

    std::vector<int64_t> vec(axis_arry1.begin(), axis_arry1.end());
    if (axis_arry1.size() == 1) {
        size_t size = 0;
        int64_t axis_array[1];
        std::copy(vec.begin(), vec.end(), axis_array);
        // expected
        auto axis = hrt::create(dt_int64, {1},
                                {reinterpret_cast<gsl::byte *>(axis_array),
                                 sizeof(axis_array)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
        auto output_ort = ortki_ReduceMax(runtime_tensor_2_ort_tensor(a),
                                          axis_array, 1, keepDims_value);
        void *ptr_ort = tensor_buffer(output_ort, &size);
        dims_t shape(tensor_rank(output_ort));
        tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
        auto expected =
            hrt::create(dt_float32, shape,
                        {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output = kernels::stackvm::reduce(
                          runtime::stackvm::reduce_op_t::max, a.impl(),
                          axis.impl(), init_value.impl(), keepDims.impl())
                          .expect("reduce_min failed");
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

    if (axis_arry1.size() == 2 && a.shape().size() >= 2) {
        int64_t axis_arr[2];
        std::copy(vec.begin(), vec.end(), axis_arr);
        // expected
        size_t size1 = 0;
        auto axis1 = hrt::create(dt_int64, {2},
                                 {reinterpret_cast<gsl::byte *>(axis_arr),
                                  sizeof(axis_arr)},
                                 true, host_runtime_tensor::pool_cpu_only)
                         .expect("create tensor failed");
        auto output_ort1 = ortki_ReduceMax(runtime_tensor_2_ort_tensor(a),
                                           axis_arr, 2, keepDims_value);
        void *ptr_ort1 = tensor_buffer(output_ort1, &size1);
        dims_t shape1(tensor_rank(output_ort1));
        tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
        auto expected1 =
            hrt::create(dt_float32, shape1,
                        {reinterpret_cast<gsl::byte *>(ptr_ort1), size1}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output1 = kernels::stackvm::reduce(
                           runtime::stackvm::reduce_op_t::max, a.impl(),
                           axis1.impl(), init_value.impl(), keepDims.impl())
                           .expect("reduce_max failed");
        runtime_tensor actual1(output1.as<tensor>().expect("as tensor failed"));

        bool result1 = is_same_tensor(expected1, actual1) ||
                       cosine_similarity_tensor(expected1, actual1);

        if (!result1) {
            std::cout << "actual ";
            print_runtime_tensor(actual1);
            std::cout << "expected ";
            print_runtime_tensor(expected1);
        }

        // compare
        EXPECT_TRUE(result1);
    }

    if (axis_arry1.size() == 3 && a.shape().size() >= 3) {
        int64_t axis_arr[3];
        std::copy(vec.begin(), vec.end(), axis_arr);
        // expected
        size_t size2 = 0;
        auto axis2 = hrt::create(dt_int64, {3},
                                 {reinterpret_cast<gsl::byte *>(axis_arr),
                                  sizeof(axis_arr)},
                                 true, host_runtime_tensor::pool_cpu_only)
                         .expect("create tensor failed");
        auto output_ort2 = ortki_ReduceMax(runtime_tensor_2_ort_tensor(a),
                                           axis_arr, 3, keepDims_value);
        void *ptr_ort2 = tensor_buffer(output_ort2, &size2);
        dims_t shape2(tensor_rank(output_ort2));
        tensor_shape(output_ort2, reinterpret_cast<int64_t *>(shape2.data()));
        auto expected2 =
            hrt::create(dt_float32, shape2,
                        {reinterpret_cast<gsl::byte *>(ptr_ort2), size2}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output2 = kernels::stackvm::reduce(
                           runtime::stackvm::reduce_op_t::max, a.impl(),
                           axis2.impl(), init_value.impl(), keepDims.impl())
                           .expect("reduce_max failed");
        runtime_tensor actual2(output2.as<tensor>().expect("as tensor failed"));

        bool result2 = is_same_tensor(expected2, actual2) ||
                       cosine_similarity_tensor(expected2, actual2);

        if (!result2) {
            std::cout << "actual ";
            print_runtime_tensor(actual2);
            std::cout << "expected ";
            print_runtime_tensor(expected2);
        }

        // compare
        EXPECT_TRUE(result2);
    }

    if (axis_arry1.size() == 4 && a.shape().size() >= 4) {
        int64_t axis_arr[4];
        std::copy(vec.begin(), vec.end(), axis_arr);
        // expected
        size_t size3 = 0;
        auto axis3 = hrt::create(dt_int64, {4},
                                 {reinterpret_cast<gsl::byte *>(axis_arr),
                                  sizeof(axis_arr)},
                                 true, host_runtime_tensor::pool_cpu_only)
                         .expect("create tensor failed");
        auto output_ort3 = ortki_ReduceMax(runtime_tensor_2_ort_tensor(a),
                                           axis_arr, 4, keepDims_value);
        void *ptr_ort3 = tensor_buffer(output_ort3, &size3);
        dims_t shape3(tensor_rank(output_ort3));
        tensor_shape(output_ort3, reinterpret_cast<int64_t *>(shape3.data()));
        auto expected3 =
            hrt::create(dt_float32, shape3,
                        {reinterpret_cast<gsl::byte *>(ptr_ort3), size3}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output3 = kernels::stackvm::reduce(
                           runtime::stackvm::reduce_op_t::max, a.impl(),
                           axis3.impl(), init_value.impl(), keepDims.impl())
                           .expect("reduce_max failed");
        runtime_tensor actual3(output3.as<tensor>().expect("as tensor failed"));

        bool result3 = is_same_tensor(expected3, actual3) ||
                       cosine_similarity_tensor(expected3, actual3);

        if (!result3) {
            std::cout << "actual ";
            print_runtime_tensor(actual3);
            std::cout << "expected ";
            print_runtime_tensor(expected3);
        }

        // compare
        EXPECT_TRUE(result3);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}