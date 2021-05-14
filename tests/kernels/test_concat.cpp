/* Copyright 2020 Canaan Inc.
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
                       std::tuple<
                           std::vector<runtime_shape_t>, // input shapes
                           runtime_shape_t, // in strides bias
                           runtime_shape_t, // out strides bias
                           size_t>> // axis
{
    // TODO:in_strides_bias add s?
public:
    void SetUp() override
    {
        auto &&[data_shapes, in_strides_bias, out_strides_bias, axis] = GetParam();

        for (size_t i = 0; i < data_shapes.size(); ++i)
        {
            in_strides.emplace_back(get_strides(data_shapes[i], in_strides_bias));
        }

        for (size_t i = 0; i < data_shapes.size(); ++i)
        {
            concat_dims.push_back(data_shapes[i][axis]);
        }

        runtime_shape_t out_shape(data_shapes[0]);
        out_shape[axis] = compute_size(concat_dims);
        output_ref = Tensor<uint32_t>(out_shape, out_strides_bias);
        output_opt = Tensor<uint32_t>(out_shape, out_strides_bias);
        this->axis = axis;
    }

    std::vector<runtime_shape_t> in_strides;
    runtime_shape_t concat_dims;
    std::vector<gsl::byte *> inputs;
    Tensor<uint32_t> output_ref, output_opt;
    size_t axis;
};

INSTANTIATE_TEST_SUITE_P(
    ConcatTestDims4,
    ConcatTest,
    testing::Combine(
        testing::Values(
            std::vector<runtime_shape_t>{ 
                runtime_shape_t { 7, 3, 4, 6 },
                runtime_shape_t { 7, 3, 4, 3 },
                runtime_shape_t { 7, 3, 4, 5 } }),
        testing::Values(runtime_shape_t { 0, 0, 0, 0 }),
        testing::Values(runtime_shape_t { 0, 0, 0, 0 }),
        testing::Values(3)));

TEST_P(ConcatTest, normal)
{
    NNCASE_UNUSED auto res1 = cpu::reference::concat(dt_float32, inputs, output_ref.gsl_ptr(),
        output_ref.shape, in_strides, output_ref.strides, axis, concat_dims);
    NNCASE_UNUSED auto res2 = cpu::optimized::concat(dt_float32, inputs, output_ref.gsl_ptr(),
        output_ref.shape, in_strides, output_ref.strides, axis, concat_dims);
    //concat(inputs, output_ref, axis, OpType::Ref);
    //concat(inputs, output_opt, axis, OpType::Opt);
    auto is_ok = output_ref == output_opt;
    if (!is_ok)
    {
        // output_all_data(inputs, output_ref, output_opt);
        ASSERT_TRUE(false);
    }
}