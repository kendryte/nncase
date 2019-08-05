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
#include "connectors.h"
#include "opcode.h"
#include <vector>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace ir
{
#define DEFINE_NODE_OPCODE(value)                                    \
    static constexpr node_opcode opcode() noexcept { return value; } \
    node_opcode runtime_opcode() const noexcept override { return value; }

    class node
    {
    public:
        node() = default;
        node(node &) = delete;
        virtual ~node();

        const std::string &name() const noexcept { return name_; }

        template <class TArg, class... TArgs>
        void name(TArg arg, TArgs... args) { name_.assign(std::forward<TArg>(arg), std::forward<TArgs>(args)...); }

        xtl::span<const input_connector> inputs() const noexcept { return input_connectors_; }
        xtl::span<const output_connector> outputs() const noexcept { return output_connectors_; }
        xtl::span<input_connector> inputs() noexcept { return input_connectors_; }
        xtl::span<output_connector> outputs() noexcept { return output_connectors_; }

        input_connector &input_at(size_t index) { return input_connectors_.at(index); }
        output_connector &output_at(size_t index) { return output_connectors_.at(index); }

        virtual node_opcode runtime_opcode() const noexcept = 0;
        virtual node_attributes attributes() const noexcept { return node_attr_none; }

    protected:
        template <class TName, class TShape>
        input_connector &add_input(TName &&name, datatype_t type, TShape &&shape)
        {
            return input_connectors_.emplace_back(*this, std::forward<TName>(name), type, std::forward<TShape>(shape));
        }

        template <class TName, class TShape>
        output_connector &add_output(TName &&name, datatype_t type, TShape &&shape, memory_type_t memory_type = mem_main)
        {
            return output_connectors_.emplace_back(*this, std::forward<TName>(name), type, std::forward<TShape>(shape), memory_type);
        }

    private:
        std::string name_;
        std::vector<input_connector> input_connectors_;
        std::vector<output_connector> output_connectors_;
    };
}
}
