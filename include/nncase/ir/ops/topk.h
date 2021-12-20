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
class NNCASE_API topk : public node
{
public:
    DEFINE_NODE_OPCODE(op_topk);

    input_connector &input() { return input_at(0); }

    // output largest values
    output_connector &output_a() { return output_at(0); }

    // output indices of largest values
    output_connector &output_b() { return output_at(1); }

    const int64_t &k() const noexcept { return k_; }
    const int32_t &axis() const noexcept { return axis_; }
    bool largest() const noexcept { return largest_; }
    bool sorted() const noexcept { return sorted_; }

    topk(datatype_t input_type, shape_t input_shape, int64_t k, int32_t axis, bool largest, bool sorted);

protected:
    bool properties_equal(node &other) const override;

private:
    int64_t k_;
    int32_t axis_;
    bool largest_;
    bool sorted_;
};
}
