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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/table_lookup.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fuse_unary.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

namespace
{
std::vector<fused_unary_op> merge_subgraph(const std::vector<fused_unary_op> &src1, const std::vector<fused_unary_op> &src2)
{
    std::vector<fused_unary_op> result = src1;
    auto arg_inc = result.size();
    for (auto &op : src2)
    {
        auto n_op = op;
        switch (n_op.opcode)
        {
        case fu_ldx:
        case fu_constant:
            break;
        case fu_unary:
            n_op.unary.input.op_id += arg_inc;
            break;
        case fu_binary:
            n_op.binary.input_a.op_id += arg_inc;
            n_op.binary.input_b.op_id += arg_inc;
            break;
        case fu_clamp:
            n_op.clamp.input.op_id += arg_inc;
            n_op.clamp.low.op_id += arg_inc;
            n_op.clamp.high.op_id += arg_inc;
            break;
        default:
            throw std::runtime_error("Invalid fused unary op");
        }

        result.emplace_back(n_op);
    }

    return result;
}
}

bool fuse_one_unary_transform::on_try_match(node &node, transform_context &context)
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

void fuse_one_unary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_u = static_cast<unary &>(*context.matched_nodes[0]);

    std::vector<fused_unary_op> subgraph;
    subgraph.emplace_back(fused_unary_op::make_ldx());
    subgraph.emplace_back(fused_unary_op::make_unary(old_u.unary_op(), { 0 }));

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph), output.type(), output.shape());
    f_u->name(output.name() + "_F");

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_one_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto bin = node_cast<binary>(node))
    {
        constant *con;
        if ((con = node_cast<constant>(bin->input_b().connection()->owner()))
            && xt::compute_size(con->output().shape()) == 1 && con->data_type() == dt_float32)
        {
            context.matched_nodes.emplace_back(bin);
            context.matched_nodes.emplace_back(con);
            context.inputs.emplace_back(&bin->input_a());
            context.outputs.emplace_back(&bin->output());
            return true;
        }
        else if ((con = node_cast<constant>(bin->input_a().connection()->owner()))
            && xt::compute_size(con->output().shape()) == 1 && con->data_type() == dt_float32)
        {
            context.matched_nodes.emplace_back(bin);
            context.matched_nodes.emplace_back(con);
            context.inputs.emplace_back(&bin->input_b());
            context.outputs.emplace_back(&bin->output());
            return true;
        }
    }

    return false;
}

void fuse_one_binary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_b = static_cast<binary &>(*context.matched_nodes[0]);
    auto &old_c = static_cast<constant &>(*context.matched_nodes[1]);

    std::vector<fused_unary_op> subgraph;
    subgraph.emplace_back(fused_unary_op::make_ldx());
    subgraph.emplace_back(fused_unary_op::make_constant(*reinterpret_cast<const float *>(old_c.data().data())));
    if (old_b.input_b().connection()->owner().runtime_opcode() == op_constant)
        subgraph.emplace_back(fused_unary_op::make_binary(old_b.binary_op(), { 0 }, { 1 }));
    else
        subgraph.emplace_back(fused_unary_op::make_binary(old_b.binary_op(), { 1 }, { 0 }));

    if (old_b.fused_activation() != value_range<float>::full())
    {
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().min));
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().max));
        subgraph.emplace_back(fused_unary_op::make_clamp({ 2 }, { 3 }, { 4 }));
    }

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph), output.type(), output.shape());
    f_u->name(output.name() + "_F");

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_two_fused_unary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto f1 = node_cast<fused_unary>(node))
    {
        if (auto f2 = try_get_direct_child<fused_unary>(*f1))
        {
            context.matched_nodes.emplace_back(f1);
            context.matched_nodes.emplace_back(f2);
            context.inputs.emplace_back(&f1->input());
            context.outputs.emplace_back(&f2->output());
            return true;
        }
    }

    return false;
}

void fuse_two_fused_unary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_f1 = static_cast<fused_unary &>(*context.matched_nodes[0]);
    auto &old_f2 = static_cast<fused_unary &>(*context.matched_nodes[1]);

    auto subgraph = concat_subgraph(old_f1.subgraph(), old_f2.subgraph());
    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph), output.type(), old_f1.input().shape());
    f_u->name(output.name() + "_F");

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_one_fused_unary_with_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (auto f = try_get_direct_parent<fused_unary>(*b, 0))
        {
            if (f->input().connection() == b->input_b().connection())
            {
                context.matched_nodes.emplace_back(f);
                context.matched_nodes.emplace_back(b);
                context.inputs.emplace_back(&f->input());
                context.inputs.emplace_back(&b->input_b());
                context.outputs.emplace_back(&b->output());
                return true;
            }
        }
        else if (auto f = try_get_direct_parent<fused_unary>(*b, 1))
        {
            if (f->input().connection() == b->input_a().connection())
            {
                context.matched_nodes.emplace_back(f);
                context.matched_nodes.emplace_back(b);
                context.inputs.emplace_back(&b->input_a());
                context.inputs.emplace_back(&f->input());
                context.outputs.emplace_back(&b->output());
                return true;
            }
        }
    }

    return false;
}

void fuse_one_fused_unary_with_binary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_f = static_cast<fused_unary &>(*context.matched_nodes[0]);
    auto &old_b = static_cast<binary &>(*context.matched_nodes[1]);

    auto subgraph = old_f.subgraph();
    auto arg_f_id = subgraph.size() - 1;
    subgraph.emplace_back(fused_unary_op::make_ldx());
    auto arg_x_id = subgraph.size() - 1;

    if (old_b.input_b().connection()->owner().runtime_opcode() == op_fused_unary)
        subgraph.emplace_back(fused_unary_op::make_binary(old_b.binary_op(), { arg_x_id }, { arg_f_id }));
    else
        subgraph.emplace_back(fused_unary_op::make_binary(old_b.binary_op(), { arg_f_id }, { arg_x_id }));

    if (old_b.fused_activation() != value_range<float>::full())
    {
        auto cur_id = subgraph.size() - 1;
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().min));
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().max));
        subgraph.emplace_back(fused_unary_op::make_clamp({ cur_id }, { cur_id + 1 }, { cur_id + 2 }));
    }

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph), output.type(), output.shape());
    f_u->name(output.name() + "_F");

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}

bool fuse_two_fused_unary_with_binary_transform::on_try_match(node &node, transform_context &context)
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

void fuse_two_fused_unary_with_binary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_f1 = static_cast<fused_unary &>(*context.matched_nodes[0]);
    auto &old_f2 = static_cast<fused_unary &>(*context.matched_nodes[1]);
    auto &old_b = static_cast<binary &>(*context.matched_nodes[2]);

    auto subgraph = merge_subgraph(old_f1.subgraph(), old_f2.subgraph());
    auto arg_a_id = old_f1.subgraph().size() - 1;
    auto arg_b_id = subgraph.size() - 1;
    subgraph.emplace_back(fused_unary_op::make_binary(old_b.binary_op(), { arg_a_id }, { arg_b_id }));

    if (old_b.fused_activation() != value_range<float>::full())
    {
        auto cur_id = subgraph.size() - 1;
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().min));
        subgraph.emplace_back(fused_unary_op::make_constant(old_b.fused_activation().max));
        subgraph.emplace_back(fused_unary_op::make_clamp({ cur_id }, { cur_id + 1 }, { cur_id + 2 }));
    }

    auto f_u = context.graph.emplace<fused_unary>(std::move(subgraph), output.type(), output.shape());
    f_u->name(output.name() + "_F");

    f_u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(f_u->output());
}
