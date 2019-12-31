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
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/fused_unary.h>
#include <hlir/ops/table_lookup.h>
#include <hlir/ops/unary.h>
#include <hlir/transforms/neutral/fuse_unary.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

namespace
{
}

bool fuse_unary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto u = node_cast<unary>(node))
    {
        context.matched_nodes.emplace_back(u);
        context.inputs.emplace_back(&u->input());
        context.outputs.emplace_back(&u->output());
        return true;
    }

    return false;
}

void fuse_unary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_u = static_cast<unary &>(*context.matched_nodes[0]);

    graph subgraph;
    auto in = subgraph.emplace<input_node>(output.type(), output.shape(), output.memory_type());
    auto u = subgraph.emplace<unary>(old_u.unary_op(), old_u.input().shape());
    auto out = subgraph.emplace<output_node>(u->output().type(), u->output().shape());
    u->input().connect(in->output());
    out->input().connect(u->output());

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph));

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_two_fused_unary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (auto f1 = try_get_direct_parent<fused_unary>(*b, 0))
        {
            if (auto f2 = try_get_direct_parent<fused_unary>(*b, 1))
            {
                if (f1->input().connection() == f2->input().connection())
                {
                    context.matched_nodes.emplace_back(f1);
                    context.matched_nodes.emplace_back(f2);
                    context.matched_nodes.emplace_back(b);
                    context.inputs.emplace_back(&f1->input());
                    context.outputs.emplace_back(&b->output());
                    return true;
                }
            }
        }
    }

    return false;
}

void fuse_two_fused_unary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_u = static_cast<unary &>(*context.matched_nodes[0]);

    graph subgraph;
    auto in = subgraph.emplace<input_node>(output.type(), output.shape(), output.memory_type());
    auto u = subgraph.emplace<unary>(old_u.unary_op(), old_u.input().shape());
    auto out = subgraph.emplace<output_node>(u->output().type(), u->output().shape());
    u->input().connect(in->output());
    out->input().connect(u->output());

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph));

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_unary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto u = node_cast<unary>(node))
    {
        context.matched_nodes.emplace_back(u);
        context.inputs.emplace_back(&u->input());
        context.outputs.emplace_back(&u->output());
        return true;
    }

    return false;
}

void fuse_unary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_u = static_cast<unary &>(*context.matched_nodes[0]);

    graph subgraph;
    auto in = subgraph.emplace<input_node>(output.type(), output.shape(), output.memory_type());
    auto u = subgraph.emplace<unary>(old_u.unary_op(), old_u.input().shape());
    auto out = subgraph.emplace<output_node>(u->output().type(), u->output().shape());
    u->input().connect(in->output());
    out->input().connect(u->output());

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph));

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}
