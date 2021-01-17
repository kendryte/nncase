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
#include <nncase/ir/graph.h>
#include <nncase/ir/visitor.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;

namespace
{
std::unordered_set<node_opcode> dontcse_ops { op_input_node, op_output_node, op_uninitialized, op_ignore_node };
}

void graph::assign_names()
{
    std::unordered_set<std::string_view> names;

    for (auto &&node : nodes_)
    {
        if (node->name().empty())
        {
            node->name(node->runtime_opcode().name);
        }

        if (names.find(node->name()) != names.end())
        {
            // needs rename
            size_t i;
            for (i = 0; names.find(node->name() + "_" + std::to_string(i)) != names.end(); i++)
            {
            }
            node->name(node->name() + "_" + std::to_string(i));
        }

        names.emplace(node->name());
    }
}

void graph::dce()
{
    std::unordered_set<node *> used_nodes;

    auto visitor = make_relay_ir_visitor([&](node &node) {
        used_nodes.emplace(&node);
    });
    visitor.visit(*this);

    auto end = std::remove_if(std::begin(nodes_), std::end(nodes_), [&](auto &node) {
        if (used_nodes.find(node.get()) == used_nodes.end())
        {
            for (auto in : node->inputs())
                in->clear_connection();
            for (auto out : node->outputs())
                out->clear_connections();
            if (node->runtime_opcode() == op_input_node)
                inputs_.erase(std::find(inputs_.begin(), inputs_.end(), static_cast<input_node *>(node.get())));
            return true;
        }

        return false;
    });
    nodes_.erase(end, std::end(nodes_));
}

std::unique_ptr<graph> graph::split_subgraph(std::span<node *> nodes)
{
    auto subgraph = std::make_unique<graph>();

    for (auto it = nodes.begin(); it != nodes.end(); ++it)
    {
        auto find_it = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &p) { return p.get() == *it; });
        if (find_it != nodes_.end())
        {
            subgraph->nodes_.emplace_back(std::move(*find_it));
            nodes_.erase(find_it);
        }
    }

    for (auto node : nodes)
    {
        for (auto in : node->inputs())
        {
            if (!in->connection())
            {
                auto inode = subgraph->emplace<input_node>(in->type(), in->shape());
                inode->name("new_input");
                in->connect(inode->output());
            }
        }

        for (auto out : node->outputs())
        {
            if (out->connections().empty())
            {
                auto onode = subgraph->emplace<output_node>(out->type(), out->shape());
                onode->name("new_output");
                out->connect(onode->input());
            }
        }
    }

    return subgraph;
}

graph &graph::add_subgraph(std::unique_ptr<graph> subgraph)
{
    return *subgraphs_.emplace_back(std::move(subgraph));
}

void graph::cse()
{
    std::unordered_set<node *> csed_nodes;

    while (true)
    {
        for (size_t i = 0; i < nodes_.size() - 1; i++)
        {
            auto &inode = nodes_[i];
            if (csed_nodes.contains(inode.get()))
                continue;

            for (size_t j = i + 1; j < nodes_.size(); j++)
            {
                auto &jnode = nodes_[j];
                if (csed_nodes.contains(jnode.get()))
                    continue;

                if (!dontcse_ops.contains(inode->runtime_opcode())
                    && inode->equals(*jnode))
                {
                    for (size_t oi = 0; oi < inode->outputs().size(); oi++)
                    {
                        auto &output = inode->output_at(oi);
                        auto inputs = jnode->output_at(oi).connections();
                        for (auto &in : std::vector<input_connector *>(inputs.begin(), inputs.end()))
                            in->connect(output);
                    }

                    csed_nodes.emplace(jnode.get());
                }
            }
        }

        if (!csed_nodes.empty())
        {
            dce();
            csed_nodes.clear();
        }
        else
        {
            break;
        }
    }
}
