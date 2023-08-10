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

class ReduceArgMaxTest : public KernelTest,
                         public ::testing::TestWithParam<
                             std::tuple<nncase::typecode_t, typecode_t, dims_t,
                                        dims_t, int64_t, int64_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode1, typecode2, l_shape, r_shape, value1, value2,
                value3] = GetParam();

        a = hrt::create(typecode1, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(a);
        axis_value = value1 > 0 ? value1 >= (int64_t)l_shape.size() ? 0 : value1
                     : -value1 > (int64_t)l_shape.size() ? 0
                                                         : value1;
        int64_t axis_array[] = {axis_value};
        axis = hrt::create(typecode2, r_shape,
                           {reinterpret_cast<gsl::byte *>(axis_array),
                            sizeof(axis_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
        keepDims_value = value2;
        int64_t keepDims_array[] = {keepDims_value};
        keepDims = hrt::create(typecode2, r_shape,
                               {reinterpret_cast<gsl::byte *>(keepDims_array),
                                sizeof(keepDims_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
        select_last_idx_value = value3;
        int64_t select_last_idx_array[] = {select_last_idx_value};
        select_last_idx =
            hrt::create(typecode2, r_shape,
                        {reinterpret_cast<gsl::byte *>(select_last_idx_array),
                         sizeof(select_last_idx_array)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor a;
    runtime_tensor axis;
    int64_t axis_value;
    runtime_tensor keepDims;
    int64_t keepDims_value;
    runtime_tensor select_last_idx;
    int64_t select_last_idx_value;
};

INSTANTIATE_TEST_SUITE_P(
    ReduceArgMax, ReduceArgMaxTest,
    testing::Combine(testing::Values(dt_float64, dt_float32/*, dt_int32, dt_int64*/),
                     testing::Values(dt_int64),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 2, 3, 4},
                                     dims_t{1, 3, 16}, dims_t{3, 16},
                                     dims_t{16}),
                     testing::Values(dims_t{1}),
                     testing::Values(-1, 0, 1, 2, 3, -2, -3, -4),
                     testing::Values(1, 0), testing::Values(1, 0)));

TEST_P(ReduceArgMaxTest, ReduceArgMax) {

    // expected
    size_t size = 0;
    auto output_ort = ortki_ArgMax(runtime_tensor_2_ort_tensor(a), axis_value,
                                   keepDims_value, select_last_idx_value);
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_int64, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::reduce_arg(runtime::stackvm::reduce_arg_op_t::arg_max,
                                     dt_int64, a.impl(), axis.impl(),
                                     keepDims.impl(), select_last_idx.impl())
            .expect("reduce_arg_max failed");
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
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}