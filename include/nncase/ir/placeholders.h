/* Copyright 2020 Canaan Inc.
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

namespace nncase::ir
{
class NNCASE_API input_node : public node
{
public:
    DEFINE_NODE_OPCODE(op_input_node);

    output_connector &output() { return output_at(0); }

    template <class TShape>
    input_node(datatype_t type, TShape &&shape)
    {
        add_output("output", type, std::forward<TShape>(shape));
    }

protected:
    bool properties_equal([[maybe_unused]] node &other) const override { return true; }
};

class NNCASE_API output_node : public node
{
public:
    DEFINE_NODE_OPCODE(op_output_node);

    input_connector &input() { return input_at(0); }

    template <class TShape>
    output_node(datatype_t type, TShape &&shape)
    {
        add_input("input", type, std::forward<TShape>(shape));
    }

protected:
    bool properties_equal([[maybe_unused]] node &other) const override { return true; }
};

class NNCASE_API ignore_node : public node
{
public:
    DEFINE_NODE_OPCODE(op_ignore_node);
    ~ignore_node() = default;

    input_connector &input() { return input_at(0); }

    template <class TShape>
    ignore_node(datatype_t type, TShape &&shape)
    {
        add_input("input", type, std::forward<TShape>(shape));
    }

protected:
    bool properties_equal([[maybe_unused]] node &other) const override { return true; }
};

class NNCASE_API uninitialized : public node
{
public:
    DEFINE_NODE_OPCODE(op_uninitialized);

    output_connector &output() { return output_at(0); }

    template <class TShape>
    uninitialized(datatype_t type, TShape &&shape)
    {
        add_output("output", type, std::forward<TShape>(shape));
    }

protected:
    bool properties_equal([[maybe_unused]] node &other) const override { return true; }
};
}
