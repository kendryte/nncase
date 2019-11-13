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
#include <ir/op_utils.h>
#include <ir/ops/binary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

binary::binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation)
    : binary_op_(binary_op), fused_activation_(fused_activation)
{
    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_output("output", dt_float32, get_binary_output_shape(input_a_shape, input_b_shape));
}

quantized_binary::quantized_binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, int32_t input_a_offset, int32_t input_a_mul, int32_t input_a_shift,
    int32_t input_b_offset, int32_t input_b_mul, int32_t input_b_shift, int32_t output_mul, int32_t output_shift, int32_t output_offset)
    : binary_op_(binary_op), input_a_offset_(input_a_offset), input_a_mul_(input_a_mul), input_a_shift_(input_a_shift), input_b_offset_(input_b_offset), input_b_mul_(input_b_mul), input_b_shift_(input_b_shift), output_mul_(output_mul), output_shift_(output_shift), output_offset_(output_offset)
{
    add_input("input_a", dt_uint8, input_a_shape);
    add_input("input_b", dt_uint8, input_b_shape);
    add_output("output", dt_uint8, get_binary_output_shape(input_a_shape, input_b_shape));
}
