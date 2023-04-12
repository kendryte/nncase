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
#include "test_util.h"
#include <gtest/gtest.h>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

class ConcatTest : public ::testing::TestWithParam<
                       std::tuple<std::vector<runtime_shape_t>, // input shapes
                                  runtime_shape_t, // in strides bias
                                  runtime_shape_t, // out strides bias
                                  size_t>>         // axis
{
  public:
    void SetUp() override {
        auto &&[data_shapes, in_strides_bias, out_strides_bias, axis] =
            GetParam();

        for (size_t i = 0; i < data_shapes.size(); ++i) {
            concat_dims.push_back(data_shapes[i][axis]);
        }

        runtime_shape_t out_shape(data_shapes[0]);
        out_shape[axis] =
            std::accumulate(concat_dims.begin(), concat_dims.end(), 0);
        output_ref = create_tensor(out_shape, out_strides_bias);
        output_opt = create_tensor(out_shape, out_strides_bias);
        this->axis = axis;

        inputs.resize(data_shapes.size());
        for (size_t i = 0; i < data_shapes.size(); ++i) {
            inputs[i] = create_input_tensor(data_shapes[i], in_strides_bias);
        }
    }

    runtime_shape_t concat_dims;
    std::vector<runtime_tensor> inputs;
    runtime_tensor output_ref, output_opt;
    size_t axis;
};

// Test name:ConcatTestDims[Dims axis]
INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims43, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3, 4, 6}, // input shape
            runtime_shape_t{7, 3, 4, 3}, runtime_shape_t{7, 3, 4, 5}}),
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // output strides bias
        testing::Values(3)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims42, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3, 3, 6}, // input shape
            runtime_shape_t{7, 3, 4, 6}, runtime_shape_t{7, 3, 5, 6}}),
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // output strides bias
        testing::Values(2)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims41, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3, 11, 6}, // input shape
            runtime_shape_t{7, 4, 11, 6}, runtime_shape_t{7, 5, 11, 6}}),
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // output strides bias
        testing::Values(1)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims40, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{5, 3, 7, 6}, // input shape
            runtime_shape_t{8, 3, 7, 6}, runtime_shape_t{3, 3, 7, 6}}),
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}), // output strides bias
        testing::Values(0)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims32, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3, 3}, // input shape
            runtime_shape_t{7, 3, 4}, runtime_shape_t{7, 3, 5}}),
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // output strides bias
        testing::Values(2)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims31, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3, 11}, // input shape
            runtime_shape_t{7, 4, 11}, runtime_shape_t{7, 5, 11}}),
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // output strides bias
        testing::Values(1)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims30, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{5, 3, 7}, // input shape
            runtime_shape_t{8, 3, 7}, runtime_shape_t{3, 3, 7}}),
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // output strides bias
        testing::Values(0)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims21, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{7, 3}, // input shape
            runtime_shape_t{7, 4}, runtime_shape_t{7, 5}}),
        testing::Values(runtime_shape_t{0, 0},
                        runtime_shape_t{3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0},
                        runtime_shape_t{3, 3}), // output strides bias
        testing::Values(1)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims20, ConcatTest,
    testing::Combine(
        testing::Values(std::vector<runtime_shape_t>{
            runtime_shape_t{5, 3}, // input shape
            runtime_shape_t{8, 3}, runtime_shape_t{3, 3}}),
        testing::Values(runtime_shape_t{0, 0},
                        runtime_shape_t{3, 3}), // input strides bias
        testing::Values(runtime_shape_t{0, 0},
                        runtime_shape_t{3, 3}), // output strides bias
        testing::Values(0)));

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims10, ConcatTest,
    testing::Combine(testing::Values(std::vector<runtime_shape_t>{
                         runtime_shape_t{5}, // input shape
                         runtime_shape_t{8}, runtime_shape_t{3}}),
                     testing::Values(runtime_shape_t{0},
                                     runtime_shape_t{3}), // input strides bias
                     testing::Values(runtime_shape_t{0},
                                     runtime_shape_t{3}), // output strides bias
                     testing::Values(0)));

void concat(std::vector<runtime_tensor> &inputs, runtime_tensor &output,
            runtime_shape_t &concat_dims, size_t axis, OpType type) {
    std::vector<runtime_shape_t> in_strides(inputs.size());
    std::vector<const gsl::byte *> inputs_v(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        in_strides[i] = inputs[i].strides();
        inputs_v[i] = get_tensor_cbegin(inputs[i]);
    }
    if (type == OpType::Ref) {
        NNCASE_UNUSED auto res1 = cpu::reference::concat(
            dt_float32, inputs_v, get_tensor_begin(output), output.shape(),
            in_strides, output.strides(), axis, concat_dims,
            default_kernel_context());
    } else if (type == OpType::Opt) {
        NNCASE_UNUSED auto res2 = cpu::optimized::concat(
            dt_float32, inputs_v, get_tensor_begin(output), output.shape(),
            in_strides, output.strides(), axis, concat_dims,
            default_kernel_context());
    } else {
        assert(false);
    }
}

TEST_P(ConcatTest, normal) {
    concat(inputs, output_ref, concat_dims, axis, OpType::Ref);
    concat(inputs, output_opt, concat_dims, axis, OpType::Opt);
    auto is_ok = is_same_tensor(output_ref, output_opt);
    if (!is_ok) {
        output_all_data(inputs, output_ref, output_opt);
        ASSERT_EQ(output_ref, output_opt);
    }
}