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

class ReduceSumTest
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

        int64_t init_value_array[] = {0}; // the sum of input's range
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
    ReduceSum, ReduceSumTest,
    testing::Combine(testing::Values(dt_float32), testing::Values(dt_int64),
                     testing::Values(dims_t{1, 2, 3, 4}),
                     testing::Values(dims_t{1}), testing::Values(0, 1),
                     testing::Values(axes_t{0}, axes_t{-1}, axes_t{-2},
                                     axes_t{-3}, axes_t{1}, axes_t{2},
                                     axes_t{3}, axes_t{2, 3}, axes_t{-2, -1},
                                     axes_t{1, 2, 3}, axes_t{-1, -2, -3},
                                     axes_t{0, 1, 2, 3},
                                     axes_t{-1, -2, -3, -4})));

TEST_P(ReduceSumTest, ReduceSum) {

    size_t axis_size = axis_arry1.size();
    if (axis_size <= a.shape().size()) {
        int64_t *axis_array = (int64_t *)malloc(axis_size * sizeof(int64_t));
        std::copy(axis_arry1.begin(), axis_arry1.end(), axis_array);
        auto axis = hrt::create(dt_int64, {axis_size},
                                {reinterpret_cast<gsl::byte *>(axis_array),
                                 axis_size * sizeof(int64_t)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
        auto output_ort = ortki_ReduceSum(runtime_tensor_2_ort_tensor(a),
                                          runtime_tensor_2_ort_tensor(axis),
                                          keepDims_value, 0);

        // expected
        size_t size = 0;
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
                          runtime::stackvm::reduce_op_t::sum, a.impl(),
                          axis.impl(), init_value.impl(), keepDims.impl())
                          .expect("reduce_sum failed");
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
        free(axis_array);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}