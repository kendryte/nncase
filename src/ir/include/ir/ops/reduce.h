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
    class reduce : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_reduce);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        reduce_op_t reduce_op() const noexcept { return reduce_op_; }
        const axis_t &axis() const noexcept { return axis_; }
        float init_value() const noexcept { return init_value_; }
        bool keep_dims() const noexcept { return keep_dims_; }

        reduce(reduce_op_t reduce_op, shape_t input_shape, axis_t axis, float init_value, bool keep_dims);

    private:
        reduce_op_t reduce_op_;
        axis_t axis_;
        float init_value_;
        bool keep_dims_;
    };

    class quantized_reduce : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_quantized_reduce);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        reduce_op_t reduce_op() const noexcept { return reduce_op_; }
        const axis_t &axis() const noexcept { return axis_; }
        float init_value() const noexcept { return init_value_; }
        bool keep_dims() const noexcept { return keep_dims_; }

        quantized_reduce(reduce_op_t reduce_op, shape_t input_shape, axis_t axis, float init_value, bool keep_dims);

    private:
        reduce_op_t reduce_op_;
        axis_t axis_;
        float init_value_;
        bool keep_dims_;
    };
}
}
