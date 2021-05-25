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

template <typename T>
void slice(const Tensor<T> &input, Tensor<T> &output,
    const runtime_shape_t &begins, const runtime_shape_t &ends,
    const runtime_axis_t &strides, OpType type)
{
    if (type == OpType::Ref)
    {
        NNCASE_UNUSED auto res = cpu::reference::slice(
            dt_float32,
            input.gsl_cptr(),
            output.gsl_ptr(),
            input.shape, 
            input.strides, 
            output.strides,
            begins, 
            ends, 
            strides);
    }
    else if (type == OpType::Opt)
    {
        NNCASE_UNUSED auto res = cpu::optimized::slice(dt_float32,
            input.gsl_cptr(),
            output.gsl_ptr(),
            input.shape, input.strides, output.strides,
            begins, ends, strides);
    }
    else
    {
        assert(false);
    }
}
class SliceTest : public ::testing::TestWithParam<
                      std::tuple<
                          runtime_shape_t, runtime_shape_t, // input shape, input strides bias
                          runtime_shape_t, runtime_shape_t, // begin end
                          runtime_shape_t, runtime_axis_t>> // out strides bias, strides
{
public:
    void SetUp() override
    {
        auto &&[data_shape, in_strides_bias, begins, ends, out_strides_bias, strides] = GetParam();

        input = Tensor<uint32_t>(data_shape, in_strides_bias);
        init_tensor_data(input);
        this->strides = strides;
        this->begins = begins;
        this->ends = ends;
        
        auto out_shape = shape_sub(begins, ends);
        for (size_t i = 0; i < out_shape.size(); ++i)
        {
            auto out_div = div(out_shape[i], strides[i]);
            out_shape[i] = (size_t)out_div.quot + (out_div.rem == 0 ? 0 : 1);
        }
        output_ref = Tensor<uint32_t>(out_shape, out_strides_bias);
        output_opt = Tensor<uint32_t>(out_shape, out_strides_bias);
    }

    void TearDown() override
    {
        delete input.data;
        delete output_ref.data;
        delete output_opt.data;
    }

    Tensor<uint32_t> input, output_ref, output_opt;
    runtime_shape_t begins, ends;
    runtime_axis_t strides;
};

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims4,
    SliceTest,
    testing::Combine(
        testing::Values(
            runtime_shape_t { 7, 4, 8, 6 }), // input shape
        testing::Values(
            runtime_shape_t { 0, 0, 0, 0 },  // input strides offset
            runtime_shape_t { 0, 0, 3, 0 },
            runtime_shape_t { 0, 3, 0, 0 },
            runtime_shape_t { 3, 0, 0, 0 },
            runtime_shape_t { 3, 3, 3, 3 }),
        testing::Values(
             runtime_shape_t { 0, 0, 0, 0 }, // begin
             runtime_shape_t { 1, 1, 1, 1 }, 
            runtime_shape_t { 2, 2, 2, 2 }),
        testing::Values(
            runtime_shape_t { 3, 3, 3, 3 }, // end
            runtime_shape_t { 3, 3, 8, 6 },
            runtime_shape_t { 7, 4, 8, 6 }),
        testing::Values(
            runtime_shape_t { 0, 0, 0, 0 }, // output strides offset
            runtime_shape_t {3, 3, 3, 3}),
        testing::Values(
            runtime_axis_t { 1, 1, 1, 1 }, // strides
            runtime_axis_t { 1, 1, 1, 3 },
            runtime_axis_t { 1, 1, 3, 1 },
            runtime_axis_t { 1, 3, 1, 1 },
            runtime_axis_t { 3, 1, 1, 1 },
            runtime_axis_t { 3, 3, 3, 3 })));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims3,
    SliceTest,
    testing::Combine(
        testing::Values(
            runtime_shape_t { 7, 8, 6 }), // input shape
        testing::Values(
            runtime_shape_t { 0, 0, 0 }, // input strides offset
            runtime_shape_t { 0, 3, 0 }, 
            runtime_shape_t { 3, 0, 0 },
            runtime_shape_t { 3, 3, 3 }),
        testing::Values(
            runtime_shape_t { 0, 0, 0 }, // begin
            runtime_shape_t { 1, 1, 1 }, 
            runtime_shape_t { 2, 2, 2 }),
        testing::Values(
            runtime_shape_t { 3, 3, 3 }, // end
            runtime_shape_t { 3, 7, 4 },
            runtime_shape_t { 7, 8, 6 }),
        testing::Values(
            runtime_shape_t { 0, 0, 0 },
            runtime_shape_t { 3, 3, 3 }), // output strides offset
        testing::Values(
            runtime_axis_t { 1, 1, 1 }, // strides
            runtime_axis_t { 1, 1, 3 },
            runtime_axis_t { 1, 3, 1 },
            runtime_axis_t { 3, 1, 1 },
            runtime_axis_t { 3, 3, 3 })));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims2,
    SliceTest,
    testing::Combine(
        testing::Values(
            runtime_shape_t { 8, 6 }), // input shape
        testing::Values(
            runtime_shape_t { 0, 0 }, // input strides offset
            runtime_shape_t { 0, 3 },
            runtime_shape_t { 3, 0 },
            runtime_shape_t { 3, 3 }),
        testing::Values(
            runtime_shape_t { 0, 0 }, // begin
            runtime_shape_t { 1, 1 },
            runtime_shape_t { 2, 2 }),
        testing::Values(
            runtime_shape_t { 3, 3 }, // end
            runtime_shape_t { 7, 4 },
            runtime_shape_t { 8, 6 }),
        testing::Values(
            runtime_shape_t { 0, 0 },
            runtime_shape_t { 3, 3 }), // output strides offset
        testing::Values(
            runtime_axis_t { 1, 1 }, // strides
            runtime_axis_t { 1, 3 },
            runtime_axis_t { 3, 1 },
            runtime_axis_t { 3, 3 })));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims1,
    SliceTest,
    testing::Combine(
        testing::Values(
            runtime_shape_t { 19 }), // input shape
        testing::Values(
            runtime_shape_t { 0 }, // input strides offset
            runtime_shape_t { 3 }), 
            testing::Values(
                runtime_shape_t { 1 }, // begin
                runtime_shape_t { 3 }),
            testing::Values(
                runtime_shape_t { 5 }, // end
                runtime_shape_t { 10 },
                runtime_shape_t { 19 }),
            testing::Values(
                runtime_shape_t { 0 },
                runtime_shape_t { 3 }), // output strides offset
            testing::Values(
                runtime_axis_t { 1 }, // strides
                runtime_axis_t { 2 },
                runtime_axis_t { 3 },
                runtime_axis_t { 4 })));


TEST_P(SliceTest, normal)
{
    slice(input, output_ref, begins, ends, strides, OpType::Ref);
    slice(input, output_opt, begins, ends, strides, OpType::Opt);
    auto is_ok = output_ref == output_opt;
    if (!is_ok)
    {
        output_all_data(input, output_ref, output_opt);
        ASSERT_EQ(output_ref, output_opt);
    }
}