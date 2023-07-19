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
          nncase::typecode_t, typecode_t, dims_t, dims_t, int64_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode1, typecode2, l_shape, r_shape, value1, value2] =
            GetParam();

        float_t a_array[] = {1, 2, 3, 4, 5, 6, 7, 8};
        a = hrt::create(
                typecode1, l_shape,
                {reinterpret_cast<gsl::byte *>(a_array), sizeof(a_array)}, true,
                host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        keepDims_value = value2;
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

        axis_value = value1;
    }

    void TearDown() override {}

  protected:
    runtime_tensor a;
    int64_t axis_value;
    int64_t keepDims_value;
    runtime_tensor keepDims;
    runtime_tensor init_value;
};

INSTANTIATE_TEST_SUITE_P(
    ReduceArgMax, ReduceMaxTest,
    testing::Combine(testing::Values(dt_float32), testing::Values(dt_int64),
                     testing::Values(dims_t{8}), testing::Values(dims_t{1}),
                     testing::Values(0, -1), testing::Values(0, 1)));

TEST_P(ReduceMaxTest, ReduceMax) {

    // expected
    size_t size = 0;
    int64_t axis_array[] = {axis_value};
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
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::reduce(runtime::stackvm::reduce_op_t::max,
                                           a.impl(), axis.impl(),
                                           init_value.impl(), keepDims.impl())
                      .expect("reduce_max failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual);

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
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}