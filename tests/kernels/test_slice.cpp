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
#include <nncase/runtime/runtime_tensor.h>

void slice(runtime_tensor &input, runtime_tensor &output,
           const runtime_shape_t &begins, const runtime_axis_t &ends,
           const runtime_axis_t &strides, OpType type) {
    if (type == OpType::Ref) {
        NNCASE_UNUSED auto res = cpu::reference::slice(
            dt_float32,
            // input.data_as<const gsl::byte>(),
            get_tensor_cbegin(input), get_tensor_begin(output), input.shape(),
            input.strides(), output.strides(), begins, ends, strides,
            default_kernel_context());
    } else if (type == OpType::Opt) {
        NNCASE_UNUSED auto res = cpu::optimized::slice(
            dt_float32, get_tensor_cbegin(input), get_tensor_begin(output),
            input.shape(), input.strides(), output.strides(), begins, ends,
            strides, default_kernel_context());
    } else {
        assert(false);
    }
}

class SliceTest
    : public ::testing::TestWithParam<std::tuple<
          runtime_shape_t, runtime_shape_t, // input shape, input strides bias
          runtime_shape_t, runtime_axis_t,  // begin end
          runtime_shape_t, runtime_axis_t>> // out strides bias, strides
{
  public:
    void SetUp() override {
        auto &&[data_shape, in_strides_bias, begins, ends, out_strides_bias,
                strides] = GetParam();

        input = create_input_tensor(data_shape, in_strides_bias);

        this->strides = strides;
        this->begins = begins;
        this->ends = ends;

        auto out_shape = shape_sub(begins, ends);
        for (size_t i = 0; i < out_shape.size(); ++i) {
            auto out_div = div(out_shape[i], strides[i]);
            out_shape[i] = (size_t)out_div.quot + (out_div.rem == 0 ? 0 : 1);
        }
        output_ref = create_tensor(out_shape, out_strides_bias);
        output_opt = create_tensor(out_shape, out_strides_bias);
    }

    runtime_tensor input, output_ref, output_opt;
    // Tensor<uint32_t> input, output_ref, output_opt;
    runtime_shape_t begins;
    runtime_axis_t ends;
    runtime_axis_t strides;
};

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims4, SliceTest,
    testing::Combine(
        testing::Values(runtime_shape_t{7, 4, 8, 6}), // input shape
        testing::Values(runtime_shape_t{0, 0, 0, 0},  // input strides offset
                        runtime_shape_t{0, 0, 3, 0},
                        runtime_shape_t{0, 3, 0, 0},
                        runtime_shape_t{3, 0, 0, 0},
                        runtime_shape_t{3, 3, 3, 3}),
        testing::Values(runtime_shape_t{0, 0, 0, 0}, // begin
                        runtime_shape_t{1, 1, 1, 1},
                        runtime_shape_t{2, 2, 2, 2}),
        testing::Values(runtime_axis_t{3, 3, 3, 3}, // end
                        runtime_axis_t{3, 3, 8, 6}, runtime_axis_t{7, 4, 8, 6}),
        testing::Values(runtime_shape_t{0, 0, 0, 0}, // output strides offset
                        runtime_shape_t{3, 3, 3, 3}),
        testing::Values(runtime_axis_t{1, 1, 1, 1}, // strides
                        runtime_axis_t{1, 1, 1, 3}, runtime_axis_t{1, 1, 3, 1},
                        runtime_axis_t{1, 3, 1, 1}, runtime_axis_t{3, 1, 1, 1},
                        runtime_axis_t{3, 3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims3, SliceTest,
    testing::Combine(
        testing::Values(runtime_shape_t{7, 8, 6}), // input shape
        testing::Values(runtime_shape_t{0, 0, 0},  // input strides offset
                        runtime_shape_t{0, 3, 0}, runtime_shape_t{3, 0, 0},
                        runtime_shape_t{3, 3, 3}),
        testing::Values(runtime_shape_t{0, 0, 0}, // begin
                        runtime_shape_t{1, 1, 1}, runtime_shape_t{2, 2, 2}),
        testing::Values(runtime_axis_t{3, 3, 3}, // end
                        runtime_axis_t{3, 7, 4}, runtime_axis_t{7, 8, 6}),
        testing::Values(runtime_shape_t{0, 0, 0},
                        runtime_shape_t{3, 3, 3}), // output strides offset
        testing::Values(runtime_axis_t{1, 1, 1},   // strides
                        runtime_axis_t{1, 1, 3}, runtime_axis_t{1, 3, 1},
                        runtime_axis_t{3, 1, 1}, runtime_axis_t{3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims2, SliceTest,
    testing::Combine(
        testing::Values(runtime_shape_t{8, 6}), // input shape
        testing::Values(runtime_shape_t{0, 0},  // input strides offset
                        runtime_shape_t{0, 3}, runtime_shape_t{3, 0},
                        runtime_shape_t{3, 3}),
        testing::Values(runtime_shape_t{0, 0}, // begin
                        runtime_shape_t{1, 1}, runtime_shape_t{2, 2}),
        testing::Values(runtime_axis_t{3, 3}, // end
                        runtime_axis_t{7, 4}, runtime_axis_t{8, 6}),
        testing::Values(runtime_shape_t{0, 0},
                        runtime_shape_t{3, 3}), // output strides offset
        testing::Values(runtime_axis_t{1, 1},   // strides
                        runtime_axis_t{1, 3}, runtime_axis_t{3, 1},
                        runtime_axis_t{3, 3})));

INSTANTIATE_TEST_SUITE_P(
    SliceTestDims1, SliceTest,
    testing::Combine(testing::Values(runtime_shape_t{19}), // input shape
                     testing::Values(runtime_shape_t{0}, // input strides offset
                                     runtime_shape_t{3}),
                     testing::Values(runtime_shape_t{1}, // begin
                                     runtime_shape_t{3}),
                     testing::Values(runtime_axis_t{5}, // end
                                     runtime_axis_t{10}, runtime_axis_t{19}),
                     testing::Values(runtime_shape_t{0},
                                     runtime_shape_t{
                                         3}),           // output strides offset
                     testing::Values(runtime_axis_t{1}, // strides
                                     runtime_axis_t{2}, runtime_axis_t{3},
                                     runtime_axis_t{4})));

TEST_P(SliceTest, normal) {
    slice(input, output_ref, begins, ends, strides, OpType::Ref);
    slice(input, output_opt, begins, ends, strides, OpType::Opt);
    auto is_ok = is_same_tensor(output_ref, output_opt);
    if (!is_ok) {
        output_all_data(input, output_ref, output_opt);
        ASSERT_EQ(output_ref, output_opt);
    }
}