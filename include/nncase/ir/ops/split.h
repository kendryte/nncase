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
    class NNCASE_API split : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_split);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        std::vector<size_t> indices_or_sections() const noexcept { return indices_or_sections_; }
        int32_t axis() const noexcept { return axis_; }
        bool is_indices() const noexcept { return is_indices_; }

        split(datatype_t type, shape_t inputs_shape, std::vector<shape_t> outputs_shape, std::vector<size_t> indices_or_sections, int32_t axis, bool is_indices);

    protected:
        bool properties_equal(node &other) const override;

    private:
        std::vector<size_t> indices_or_sections_;
        int32_t axis_;
        bool is_indices_;
    };
}
