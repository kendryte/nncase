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
class NNCASE_API reduce_prod : public node
{
public:
    DEFINE_NODE_OPCODE(op_reduce_prod);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    const axis_t &axis() const noexcept { return axis_; }
    bool keep_dims() const noexcept { return keep_dims_; }

    reduce_prod(shape_t input_shape, axis_t axis, bool keep_dims);

protected:
    bool properties_equal(node &other) const override;

private:
    axis_t axis_;
    bool keep_dims_;
};
}
