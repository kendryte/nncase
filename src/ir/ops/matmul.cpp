/* Copyright 2019 Canaan Inc.
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
#include <ir/ops/matmul.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

matmul::matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<float, 1> bias, value_range<float> fused_activation)
    : bias_(std::move(bias)), fused_activation_(fused_activation)
{
    if (input_a_shape.size() != 2 || input_b_shape.size() != 2)
        throw std::invalid_argument("inputs must be 2 rank");
    if (input_a_shape[1] != input_b_shape[0])
        throw std::invalid_argument("input a's cols must be equal to input b's rows");

    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_output("output", dt_float32, shape_t { input_a_shape[0], input_b_shape[1] });
}

quantized_matmul::quantized_matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<int32_t, 1> bias, int32_t input_a_offset, int32_t input_b_offset, int32_t output_mul, int32_t output_shift, int32_t output_offset)
    : bias_(std::move(bias)), input_a_offset_(input_a_offset), input_b_offset_(input_b_offset), output_mul_(output_mul), output_shift_(output_shift), output_offset_(output_offset)
{
    if (input_a_shape.size() != 2 || input_b_shape.size() != 2)
        throw std::invalid_argument("inputs must be 2 rank");
    if (input_a_shape[1] != input_b_shape[0])
        throw std::invalid_argument("input a's cols must be equal to input b's rows");

    add_input("input_a", dt_uint8, input_a_shape);
    add_input("input_b", dt_uint8, input_b_shape);
    add_output("output", dt_uint8, shape_t { input_a_shape[0], input_b_shape[1] });
}
