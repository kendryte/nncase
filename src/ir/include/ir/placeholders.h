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
#include "node.h"

namespace nncase
{
namespace ir
{
    class input_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_input_node);

        output_connector &output() { return output_at(0); }

        template <class TShape>
        input_node(datatype_t type, TShape &&shape, memory_type_t memory_type = mem_main)
        {
            add_output("output", type, std::forward<TShape>(shape), memory_type);
        }
    };

    class output_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_output_node);

        input_connector &input() { return input_at(0); }

        template <class TShape>
        output_node(datatype_t type, TShape &&shape)
        {
            add_input("input", type, std::forward<TShape>(shape));
        }
    };

    class ignore_node : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_ignore_node);

        input_connector &input() { return input_at(0); }

        template <class TShape>
        ignore_node(datatype_t type, TShape &&shape)
        {
            add_input("input", type, std::forward<TShape>(shape));
        }

        node_attributes attributes() const noexcept override { return node_attr_action; }
    };
}
}
