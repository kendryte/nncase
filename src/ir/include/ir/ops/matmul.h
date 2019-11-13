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
    class matmul : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_matmul);

        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }

        const xt::xtensor<float, 1> &bias() const noexcept { return bias_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<float, 1> bias, value_range<float> fused_activation);

    private:
        xt::xtensor<float, 1> bias_;
        value_range<float> fused_activation_;
    };

    class quantized_matmul : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_quantized_matmul);

        input_connector &input_a() { return input_at(0); }
        input_connector &input_b() { return input_at(1); }
        output_connector &output() { return output_at(0); }
        int32_t input_a_offset() const noexcept { return input_a_offset_; }
        int32_t input_b_offset() const noexcept { return input_b_offset_; }
        int32_t output_mul() const noexcept { return output_mul_; }
        int32_t output_shift() const noexcept { return output_shift_; }
        int32_t output_offset() const noexcept { return output_offset_; }

        const xt::xtensor<int32_t, 1> &bias() const noexcept { return bias_; }

        quantized_matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<int32_t, 1> bias, int32_t input_a_offset, int32_t input_b_offset, int32_t output_mul, int32_t output_shift, int32_t output_offset);

    private:
        xt::xtensor<int32_t, 1> bias_;
        int32_t input_a_offset_;
        int32_t input_b_offset_;
        int32_t output_mul_;
        int32_t output_shift_;
        int32_t output_offset_;
    };
}
}
