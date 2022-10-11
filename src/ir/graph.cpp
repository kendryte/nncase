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
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/call.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;

namespace
{
std::unordered_set<node_opcode> dontcse_ops { op_input_node, op_output_node, op_uninitialized, op_constant, op_ignore_node };
std::unordered_set<char> char_need_escape = { '/', ':' };

void add_reachable_graphs(graph &root, std::vector<graph *> &graphs)
{
    graphs.emplace_back(&root);
    std::unordered_set<graph *> subgraphs;
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (auto c = node_cast<call>(node))
            subgraphs.emplace(&c->target());
    });
    visitor.visit(root);
    for (auto &g : subgraphs)
        add_reachable_graphs(*g, graphs);
}
}

graph::graph() noexcept
    : graph(runtime::stackvm::stackvm_module_type)
{
}

std::string graph::escaped_name() const noexcept
{
    auto escaped_name = name();
    for (auto &c : escaped_name)
    {
        if (char_need_escape.contains(c))
            c = '_';
    }

    return escaped_name;
}

std::string node::escaped_name() const noexcept
{
    auto escaped_name = name();
    for (auto &c : escaped_name)
    {
        if (char_need_escape.contains(c))
            c = '_';
    }

    return escaped_name;
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
                ;
            node->name(node->name() + "_" + std::to_string(i));
        }

        names.emplace(node->name());
    }
}

void graph::dce()
{
    std::unordered_set<node *> used_nodes;

    for (auto it = outputs_.begin(); it != outputs_.end();)
    {
        if (!(*it)->input().connection())
        {
            nodes_.erase(std::find_if(nodes_.begin(), nodes_.end(), [it](std::unique_ptr<node> &node) { return node.get() == *it; }));
            it = outputs_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    auto visitor = make_relay_ir_visitor([&](node &node) { used_nodes.emplace(&node); });
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

split_graph_result graph::split_subgraph(std::span<node *const> nodes, bool reorder_input)
{
    split_graph_result result;
    result.subgraph = std::make_unique<graph>(nodes.front()->module_type());

    // 1. Erase nodes
    std::unordered_set<node *> subgraph_nodes;
    for (auto it = nodes.begin(); it != nodes.end(); ++it)
    {
        auto find_it = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &p) { return p.get() == *it; });
        if (find_it != nodes_.end())
        {
            subgraph_nodes.emplace(find_it->get());
            result.subgraph->nodes_.emplace_back(std::move(*find_it));
            nodes_.erase(find_it);
        }
    }

#define ADD_INODE                                                               \
    auto inode = result.subgraph->emplace<input_node>(in->type(), in->shape()); \
    inode->name(in->connection()->owner().name());                              \
    inode->module_type(in->owner().module_type());                              \
    result.inputs.emplace(inode, in->connection());                             \
    inputs.emplace(in->connection(), inode);                                    \
    in->connect(inode->output());

#define ADD_ONODE                                                                  \
    auto onode = result.subgraph->emplace<output_node>(out->type(), out->shape()); \
    onode->name(out->owner().name());                                              \
    onode->module_type(out->owner().module_type());                                \
                                                                                   \
    for (auto in : dup(conns))                                                     \
    {                                                                              \
        if (!subgraph_nodes.contains(&in->owner()))                                \
        {                                                                          \
            result.outputs[onode].emplace_back(in);                                \
            in->clear_connection();                                                \
        }                                                                          \
    }                                                                              \
                                                                                   \
    out->connect(onode->input());

    // 2. Find in/out connectors
    std::unordered_set<output_connector *> outputs;
    std::unordered_map<output_connector *, input_node *> inputs;
    std::vector<input_connector *> graph_inputs;
    std::vector<input_connector *> remained_inputs;
    std::vector<size_t> input_order;
    std::vector<output_connector *> graph_outputs;
    std::vector<size_t> output_order;
    for (auto node : nodes)
    {
        for (auto in : node->inputs())
        {
            if (!subgraph_nodes.contains(&in->connection()->owner()))
            {
                if (outputs.emplace(in->connection()).second)
                {
                    if (reorder_input && node_cast<input_node>(in->connection()->owner()))
                    {
                        graph_inputs.push_back(in);
                        input_order.push_back(std::distance(inputs_.begin(), std::find(inputs_.begin(), inputs_.end(), node_cast<input_node>(in->connection()->owner()))));
                    }
                    else
                    {
                        ADD_INODE
                    }
                }
                else
                {
                    if (reorder_input && node_cast<input_node>(in->connection()->owner()))
                        remained_inputs.push_back(in);
                    else
                        in->connect(inputs.at(in->connection())->output());
                }
            }
        }

        for (auto out : node->outputs())
        {
            auto conns = out->connections();
            if (std::any_of(conns.begin(), conns.end(), [&](input_connector *in) { return !subgraph_nodes.contains(&in->owner()); }))
            {
                auto it = std::find_if(conns.begin(), conns.end(), [&](input_connector *in) { return node_cast<output_node>(in->owner()); });
                if (it != conns.end())
                {
                    graph_outputs.push_back(out);
                    output_order.push_back(std::distance(outputs_.begin(), std::find(outputs_.begin(), outputs_.end(), node_cast<output_node>((*it)->owner()))));
                }
                else
                {
                    ADD_ONODE
                }
            }
        }
    }

    auto sort_indexes = [](const std::vector<size_t> &v) {
        std::vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        stable_sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

        return idx;
    };

    auto ordered_in_indexes = sort_indexes(input_order);
    for (auto idx : ordered_in_indexes)
    {
        auto in = graph_inputs[idx];
        ADD_INODE
    }
    for (auto &in : remained_inputs)
        in->connect(inputs.at(in->connection())->output());

    auto ordered_out_indexes = sort_indexes(output_order);
    for (auto idx : ordered_out_indexes)
    {
        auto out = graph_outputs[idx];
        auto conns = out->connections();
        ADD_ONODE
    }

#undef ADD_ONODE

    return result;
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
                    && inode->module_type() == jnode->module_type()
                    && inode->equals(*jnode))
                {
                    for (size_t oi = 0; oi < inode->outputs().size(); oi++)
                    {
                        auto &output = inode->output_at(oi);
                        auto inputs = dup(jnode->output_at(oi).connections());
                        for (auto in : inputs)
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

std::vector<graph *> graph::reachable_graphs() noexcept
{
    std::vector<graph *> graphs;
    add_reachable_graphs(*this, graphs);
    return graphs;
}
