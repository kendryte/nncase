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
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/runtime_op_utility.h>

class CopyTest : public ::testing::TestWithParam<
                     std::tuple<runtime_shape_t,  // shape
                                runtime_shape_t,  // src strides bias
                                runtime_shape_t>> // dest strides bias
{
  public:
    void SetUp() override {
        auto &&[shape, in_strides_bias, out_strides_bias] = GetParam();
        input = create_input_tensor(shape, in_strides_bias);

        output_ref = create_tensor(shape, out_strides_bias);
        output_opt = create_tensor(shape, out_strides_bias);
    }

    runtime_tensor input, output_ref, output_opt;
};

INSTANTIATE_TEST_SUITE_P(
    CopyTestD1, CopyTest,
    testing::Combine(testing::Values(runtime_shape_t{11}), // shape
                     testing::Values(runtime_shape_t{0},   // src strides bias
                                     runtime_shape_t{1}),
                     testing::Values(runtime_shape_t{0}))); // dest strides bias

INSTANTIATE_TEST_SUITE_P(
    CopyTestD2, CopyTest,
    testing::Combine(
        testing::Values(runtime_shape_t{3, 5}), // shape
        testing::Values(runtime_shape_t{1, 0},  // src strides bias
                        runtime_shape_t{0, 1}, runtime_shape_t{0, 0}),
        testing::Values(runtime_shape_t{0, 0}))); // dest strides bias

INSTANTIATE_TEST_SUITE_P(
    CopyTestD3, CopyTest,
    testing::Combine(
        testing::Values(runtime_shape_t{3, 5, 4}), // shape
        testing::Values(runtime_shape_t{0, 0, 0},  // src strides bias
                        runtime_shape_t{1, 0, 0}, runtime_shape_t{0, 1, 0},
                        runtime_shape_t{0, 0, 1}),
        testing::Values(runtime_shape_t{0, 0, 0}))); // dest strides bias

INSTANTIATE_TEST_SUITE_P(
    CopyTestD4, CopyTest,
    testing::Combine(
        testing::Values(runtime_shape_t{3, 5, 4, 7}), // shape
        testing::Values(runtime_shape_t{0, 0, 0, 0},  // src strides bias
                        runtime_shape_t{1, 0, 0, 0},
                        runtime_shape_t{0, 1, 0, 0},
                        runtime_shape_t{0, 0, 1, 0},
                        runtime_shape_t{0, 0, 0, 1}),
        testing::Values(runtime_shape_t{0, 0, 0, 0}, // dest strides bias
                        runtime_shape_t{1, 0, 0, 0},
                        runtime_shape_t{0, 1, 0, 0},
                        runtime_shape_t{0, 0, 1, 0},
                        runtime_shape_t{0, 0, 0, 1})));

INSTANTIATE_TEST_SUITE_P(
    CopyTest0, CopyTest,
    testing::Combine(testing::Values(runtime_shape_t{1, 1, 1, 1}), // shape
                     testing::Values(runtime_shape_t{0, 0, 0, 0}
                                     // src strides bias
                                     ),
                     testing::Values(runtime_shape_t{0, 0, 0, 0}
                                     // dest strides bias
                                     )));

void copy(runtime_tensor &input, runtime_tensor &output, OpType type) {
    if (type == OpType::Ref) {
        NNCASE_UNUSED auto res = cpu::reference::copy(
            dt_float32, get_tensor_cbegin(input), get_tensor_begin(output),
            input.shape(), input.strides(), output.strides(),
            default_kernel_context());
    } else if (type == OpType::Opt) {
        NNCASE_UNUSED auto res = kernels::copy(
            dt_float32, get_tensor_cbegin(input), get_tensor_begin(output),
            input.shape(), input.strides(), output.strides());
    } else {
        assert(false);
    }
}

TEST_P(CopyTest, normal) {
    copy(input, output_ref, OpType::Ref);
    copy(input, output_opt, OpType::Opt);
    auto is_ok = is_same_tensor(output_ref, output_opt);
    if (!is_ok) {
        output_all_data(input, output_ref, output_opt);
        ASSERT_EQ(output_ref, output_opt);
    }
}