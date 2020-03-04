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
#include <hlir/graph.h>
#include <hlir/visitor.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::hlir;

void graph::assign_names()
{
    std::unordered_set<std::string_view> names;

    for (auto &&node : nodes_)
    {
        size_t i = 0;
        while (node->name().empty() || names.find(node->name()) != names.end())
            node->name(std::string(node_opcode_names(node->runtime_opcode())) + "_" + std::to_string(i++));
        names.emplace(node->name());
    }
}

void graph::collect()
{
    std::unordered_set<node *> used_nodes;

    auto visitor = make_relay_ir_visitor([&](node &node) {
        used_nodes.emplace(&node);
    });
    visitor.visit(*this);

    auto end = std::remove_if(std::begin(nodes_), std::end(nodes_), [&](auto &node) {
        if (used_nodes.find(node.get()) == used_nodes.end())
        {
            for (auto &in : node->inputs())
                in.clear_connection();
            for (auto &out : node->outputs())
                out.clear_connections();
            if (node->runtime_opcode() == op_input_node)
                inputs_.erase(std::find(inputs_.begin(), inputs_.end(), static_cast<input_node *>(node.get())));
            return true;
        }

        return false;
    });
    nodes_.erase(end, std::end(nodes_));
}

void graph::compile(hlir_compile_context &context)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        node.compile(context);
    });
    visitor.visit(*this);

    for (auto &&in : context.h_inputs)
    {
        auto out = context.h_outputs.find(in.first->connection());
        if (out != context.h_outputs.end())
            in.second->connect(*out->second);
    }

    for (auto &&out : context.h_outputs)
    {
        for (auto &&conn : out.first->connections())
        {
            auto in = context.h_inputs.find(conn);
            if (in != context.h_inputs.end())
                out.second->connect(*in->second);
        }
    }
}

void graph::flatten_subgraph(hlir_compile_context &context,
    const std::unordered_map<hlir::output_connector *, llir::output_connector *> &inputs,
    const std::unordered_map<hlir::input_connector *, llir::input_connector *> &outputs)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (node.runtime_opcode() != op_input_node
            && node.runtime_opcode() != op_output_node)
            node.compile(context);
    });
    visitor.visit(*this);

    for (auto &&in : context.h_inputs)
    {
        auto out = context.h_outputs.find(in.first->connection());
        if (out != context.h_outputs.end())
        {
            in.second->connect(*out->second);
        }
        else
        {
            auto out = inputs.find(in.first->connection());
            if (out != inputs.end())
                in.second->connect(*out->second);
        }
    }

    for (auto &&out : context.h_outputs)
    {
        for (auto &&conn : out.first->connections())
        {
            auto in = context.h_inputs.find(conn);
            if (in != context.h_inputs.end())
            {
                out.second->connect(*in->second);
            }
            else
            {
                auto in = outputs.find(conn);
                if (in != outputs.end())
                    out.second->connect(*in->second);
            }
        }
    }
}
