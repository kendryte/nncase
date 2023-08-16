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

#define TEST_CASE_NAME "test_reduce_window2D"

class ReduceWindow2DTest : public KernelTest,
                           public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()
        auto typecode = GetDataType("lhs_type");
        auto l_shape = GetShapeArray("lhs_shape");
        auto value1 = GetAxesArray("dilations");
        auto value2 = GetAxesArray("filter");
        auto value3 = GetAxesArray("stride");
        auto value4 = GetAxesArray("onnxPads");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        dilations_value = value1;
        filter_value = value2;
        stride_value = value3;
        onnxPads_value = value4;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    axes_t dilations_value;
    axes_t filter_value;
    axes_t stride_value;
    axes_t onnxPads_value;
};

INSTANTIATE_TEST_SUITE_P(ReduceWindow2D, ReduceWindow2DTest,
                         testing::Combine(testing::Range(0, 2)));

TEST_P(ReduceWindow2DTest, ReduceWindow2D) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    size_t dilations_size = dilations_value.size();
    int64_t *dilations = (int64_t *)malloc(dilations_size * sizeof(int64_t));
    std::copy(dilations_value.begin(), dilations_value.end(), dilations);

    size_t filter_size = filter_value.size();
    int64_t *filter = (int64_t *)malloc(filter_size * sizeof(int64_t));
    std::copy(filter_value.begin(), filter_value.end(), filter);

    size_t stride_size = stride_value.size();
    int64_t *stride = (int64_t *)malloc(stride_size * sizeof(int64_t));
    std::copy(stride_value.begin(), stride_value.end(), stride);

    size_t onnxPads_size = onnxPads_value.size();
    int64_t *onnxPads = (int64_t *)malloc(onnxPads_size * sizeof(int64_t));
    std::copy(onnxPads_value.begin(), onnxPads_value.end(), onnxPads);
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
        hrt::create(dt_int64, {dilations_size},
                    {reinterpret_cast<gsl::byte *>(dilations),
                     dilations_size * sizeof(int64_t)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto filter_tensor = hrt::create(dt_int64, {filter_size},
                                     {reinterpret_cast<gsl::byte *>(filter),
                                      filter_size * sizeof(int64_t)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");

    auto stride_tensor = hrt::create(dt_int64, {stride_size},
                                     {reinterpret_cast<gsl::byte *>(stride),
                                      stride_size * sizeof(int64_t)},
                                     true, host_runtime_tensor::pool_cpu_only)
                             .expect("create tensor failed");

    auto onnxPads_tensor = hrt::create(dt_int64, {onnxPads_size},
                                       {reinterpret_cast<gsl::byte *>(onnxPads),
                                        onnxPads_size * sizeof(int64_t)},
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
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(lhs_shape, j)
    FOR_LOOP(dilations, k)
    FOR_LOOP(filter, l)
    FOR_LOOP(stride, m)
    FOR_LOOP(onnxPads, n)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(lhs_shape, j)
    SPLIT_ELEMENT(dilations, k)
    SPLIT_ELEMENT(filter, l)
    SPLIT_ELEMENT(stride, m)
    SPLIT_ELEMENT(onnxPads, n)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}