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

class ReduceWindow2DTest
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

INSTANTIATE_TEST_SUITE_P(ReduceWindow2D, ReduceWindow2DTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(ReduceWindow2DTest, ReduceWindow2D) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    int64_t dilations[] = {1, 1};
    int64_t filter[] = {3, 3};
    int64_t stride[] = {1, 1};
    int64_t onnxPads[] = {1, 1, 1, 1};
    auto output_ort = ortki_MaxPool(l_ort, "NOTSET", 0, dilations, 2, filter, 2,
                                    onnxPads, 4, 0, stride, 2);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(tensor_seq_get_value(output_ort, 0), &size);
    dims_t shape(tensor_rank(tensor_seq_get_value(output_ort, 0)));
    tensor_shape(tensor_seq_get_value(output_ort, 0),
                 reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto dilations_tensor =
        hrt::create(
            dt_int64, {2},
            {reinterpret_cast<gsl::byte *>(dilations), sizeof(dilations)}, true,
            host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto filter_tensor =
        hrt::create(dt_int64, {2},
                    {reinterpret_cast<gsl::byte *>(filter), sizeof(filter)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto stride_tensor =
        hrt::create(dt_int64, {2},
                    {reinterpret_cast<gsl::byte *>(stride), sizeof(stride)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto onnxPads_tensor =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(onnxPads), sizeof(onnxPads)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    float_t init_value[] = {0.0f};
    auto init_value_tensor =
        hrt::create(
            dt_float32, {1},
            {reinterpret_cast<gsl::byte *>(init_value), sizeof(init_value)},
            true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    bool ceil_mode_value[] = {false};
    auto ceil_mode_value_tensor =
        hrt::create(dt_boolean, {1},
                    {reinterpret_cast<gsl::byte *>(ceil_mode_value),
                     sizeof(ceil_mode_value)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    bool count_include_pad[] = {false};
    auto count_include_pad_tensor =
        hrt::create(dt_boolean, {1},
                    {reinterpret_cast<gsl::byte *>(count_include_pad),
                     sizeof(count_include_pad)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output = kernels::stackvm::reduce_window2d(
                      runtime::stackvm::reduce_op_t::max, input.impl(),
                      init_value_tensor.impl(), filter_tensor.impl(),
                      stride_tensor.impl(), onnxPads_tensor.impl(),
                      dilations_tensor.impl(), ceil_mode_value_tensor.impl(),
                      count_include_pad_tensor.impl())
                      .expect("reduce_window_max failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    //    EXPECT_FALSE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}