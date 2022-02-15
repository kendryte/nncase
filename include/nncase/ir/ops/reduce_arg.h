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
#pragma once
#include "../node.h"

namespace nncase::ir
{
class NNCASE_API reduce_arg : public node
{
public:
    DEFINE_NODE_OPCODE(op_reduce_arg);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    reduce_arg_op_t reduce_arg_op() const noexcept { return reduce_arg_op_; }
    int32_t axis() const noexcept { return axis_; }
    bool keep_dims() const noexcept { return keep_dims_; }
    bool select_last_index() const noexcept { return select_last_index_; }

    reduce_arg(reduce_arg_op_t op, datatype_t input_type, shape_t input_shape, datatype_t output_type, int32_t axis, bool keep_dims = true, bool select_last_index = false);

protected:
    bool properties_equal(node &other) const override;

private:
    reduce_arg_op_t reduce_arg_op_;
    int32_t axis_;
    bool keep_dims_;
    bool select_last_index_;
};
}
