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
#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class binary : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_binary);

        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }

        binary_op_t binary_op() const noexcept { return binary_op_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation);

    private:
        binary_op_t binary_op_;
        value_range<float> fused_activation_;
    };

    class quantized_binary : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_quantized_binary);

        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }
        int32_t input_a_offset() const noexcept { return input_a_offset_; }
        int32_t input_a_mul() const noexcept { return input_a_mul_; }
        int32_t input_a_shift() const noexcept { return input_a_shift_; }
        int32_t input_b_offset() const noexcept { return input_b_offset_; }
        int32_t input_b_mul() const noexcept { return input_b_mul_; }
        int32_t input_b_shift() const noexcept { return input_b_shift_; }
        int32_t output_mul() const noexcept { return output_mul_; }
        int32_t output_shift() const noexcept { return output_shift_; }
        int32_t output_offset() const noexcept { return output_offset_; }

        binary_op_t binary_op() const noexcept { return binary_op_; }

        quantized_binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, int32_t input_a_offset, int32_t input_a_mul, int32_t input_a_shift,
            int32_t input_b_offset, int32_t input_b_mul, int32_t input_b_shift, int32_t output_mul, int32_t output_shift, int32_t output_offset);

    private:
        binary_op_t binary_op_;
        int32_t input_a_offset_;
        int32_t input_a_mul_;
        int32_t input_a_shift_;
        int32_t input_b_offset_;
        int32_t input_b_mul_;
        int32_t input_b_shift_;
        int32_t output_mul_;
        int32_t output_shift_;
        int32_t output_offset_;
    };
}
}
