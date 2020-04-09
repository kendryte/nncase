/* Copyright 2019-2020 Canaan Inc.
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
#include <llir/graph.h>
#include <unordered_map>
#include <vector>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace hlir
{
#define DEFINE_NODE_OPCODE(value)                                    \
    static constexpr node_opcode opcode() noexcept { return value; } \
    node_opcode runtime_opcode() const noexcept override { return value; }

    struct hlir_compile_context
    {
        llir::graph graph;
        std::unordered_map<input_connector *, llir::input_connector *> h_inputs;
        std::unordered_map<output_connector *, llir::output_connector *> h_outputs;
        std::unordered_map<llir::output_connector *, output_connector *> l_outputs;

        void add_input(input_connector &h, llir::input_connector &l)
        {
            h_inputs.emplace(&h, &l);
        }

        void add_output(output_connector &h, llir::output_connector &l)
        {
            h_outputs.emplace(&h, &l);
            l_outputs.emplace(&l, &h);
        }
    };

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
        node_attributes attributes() const noexcept { return attributes_; }
        void attributes(node_attributes value) noexcept { attributes_ = value; }
        virtual void compile(hlir_compile_context &context);

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
        node_attributes attributes_ = node_attributes::node_attr_none;
        std::vector<input_connector> input_connectors_;
        std::vector<output_connector> output_connectors_;
    };
}
}
