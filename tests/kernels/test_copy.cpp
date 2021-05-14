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

class CopyTest : public ::testing::TestWithParam<
                      std::tuple<
                          runtime_shape_t, // shape
                          runtime_shape_t, // input strides bias
                          runtime_shape_t>> // output strides bias
{
public:
    void SetUp() override
    {
        auto &&[shape, in_strides_bias, out_strides_bias] = GetParam();
        input = Tensor<uint32_t>(shape, in_strides_bias);
        init_tensor_data(input);

        output_ref = Tensor<uint32_t>(shape, out_strides_bias);
        output_opt = Tensor<uint32_t>(shape, out_strides_bias);
    }

    void TearDown() override
    {
    }

    Tensor<uint32_t> input, output_ref, output_opt;
};

INSTANTIATE_TEST_SUITE_P(
    CopyTestDims4,
    CopyTest,
    testing::Combine(
        testing::Values(
            runtime_shape_t {}),
        testing::Values(
            runtime_shape_t {}),
        testing::Values(
            runtime_shape_t {})));

template<typename T>
void copy(const Tensor<T>& input, Tensor<T>& output, OpType type)
{
    if (type == OpType::Ref)
    {
        NNCASE_UNUSED auto res = cpu::reference::copy(
            dt_float32,
            input.gsl_cptr(),
            output.gsl_ptr(),
            input.shape,
            input.strides,
            output.strides);
    }
    else if (type == OpType::Opt)
    {
        //NNCASE_UNUSED auto res = cpu::optimized::copy(
        //    dt_float32,
        //    input.gsl_cptr(),
        //    output.gsl_ptr(),
        //    input.shape,
        //    input.strides,
        //    output.strides);
    }
    else
    {
        assert(false);
    }
}

TEST_P(CopyTest, normal)
{
    copy(input, output_ref, OpType::Ref);
    copy(input, output_opt, OpType::Opt);
    auto is_ok = output_ref == output_opt;
    if (!is_ok)
    {
        output_all_data(input, output_ref, output_opt);
        ASSERT_TRUE(false);
    }
}
