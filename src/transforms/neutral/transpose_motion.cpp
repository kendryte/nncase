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
#include <ir/ops/binary.h>
#include <ir/ops/concat.h>
#include <ir/ops/constant.h>
#include <ir/ops/pad.h>
#include <ir/ops/reduce.h>
#include <ir/ops/transpose.h>
#include <ir/ops/unary.h>
#include <ir/visitor.h>
#include <transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool transpose_binary_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto bin = node_cast<binary>(node))
    {
        transpose *tp1, *tp2;
        if ((tp1 = node_cast<transpose>(bin->input_a().connection()->owner()))
            && (tp2 = node_cast<transpose>(bin->input_b().connection()->owner())))
        {
            if (tp1->perm() == tp2->perm())
            {
                context.inputs.emplace_back(&tp1->input());
                context.inputs.emplace_back(&tp2->input());
                context.outputs.emplace_back(&bin->output());

                context.matched_nodes.emplace_back(tp1);
                context.matched_nodes.emplace_back(tp2);
                context.matched_nodes.emplace_back(bin);
                return true;
            }
        }
    }

    return false;
}

void transpose_binary_motion_transform::process(transform_context &context)
{
    auto &output_a = *context.inputs[0]->connection();
    auto &output_b = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_bin = static_cast<binary &>(*context.matched_nodes[2]);

    auto bin = context.graph.emplace<binary>(old_bin.binary_op(), output_a.shape(), output_b.shape(), old_bin.fused_activation());
    auto tp = context.graph.emplace<transpose>(bin->output().type(), bin->output().shape(), old_tp.perm());
    tp->input().connect(bin->output());

    bin->input_a().connect(output_a);
    bin->input_b().connect(output_b);

    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool transpose_constant_binary_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto bin = node_cast<binary>(node))
    {
        transpose *tp;
        constant *con;
        if ((tp = node_cast<transpose>(bin->input_a().connection()->owner()))
            && (con = node_cast<constant>(bin->input_b().connection()->owner()))
            && con->output().shape().size() == 1)
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&bin->output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(con);
            context.matched_nodes.emplace_back(bin);
            return true;
        }
        else if ((tp = node_cast<transpose>(bin->input_b().connection()->owner()))
            && (con = node_cast<constant>(bin->input_a().connection()->owner()))
            && con->output().shape().size() == 1)
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&bin->output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(con);
            context.matched_nodes.emplace_back(bin);
            return true;
        }
    }

    return false;
}

void transpose_constant_binary_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_con = static_cast<constant &>(*context.matched_nodes[1]);
    auto &old_bin = static_cast<binary &>(*context.matched_nodes[2]);
    auto &old_perm = old_tp.perm();
    auto expand_dim = old_perm.size() - old_perm[old_perm.size() - 1] - 1;

    auto con_shape = old_con.output().shape();
    if (con_shape[0] != 1)
    {
        for (size_t i = 0; i < expand_dim; i++)
            con_shape.push_back(1);
    }

    auto con = context.graph.emplace<constant>(old_con.output().type(), con_shape, old_con.data());
    auto bin = context.graph.emplace<binary>(old_bin.binary_op(), output.shape(), con->output().shape(), old_bin.fused_activation());
    auto tp = context.graph.emplace<transpose>(bin->output().type(), bin->output().shape(), old_tp.perm());
    tp->input().connect(bin->output());

    bin->input_a().connect(output);
    bin->input_b().connect(con->output());

    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

// Transpose (perm = p1)
//     |
//     v
// concat (perm = p2)
//
// p1[p2[i]] == i

bool transpose_concat_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_concat)
    {
        auto &con = static_cast<concat &>(node);

        context.matched_nodes.emplace_back(&con);
        context.outputs.emplace_back(&con.output());

        axis_t perm;
        for (auto &&conn : con.inputs())
        {
            if (conn.connection()->owner().runtime_opcode() == op_transpose)
            {
                auto &tp = static_cast<transpose &>(conn.connection()->owner());

                if (perm.empty())
                    perm = tp.perm();

                if (perm != tp.perm())
                    return false;

                context.inputs.emplace_back(&tp.input());
                context.matched_nodes.emplace_back(&tp);
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    return false;
}

void transpose_concat_motion_transform::process(transform_context &context)
{
    auto &con = static_cast<concat &>(*context.matched_nodes[0]);
    auto &perm = static_cast<transpose &>(*context.matched_nodes[1]).perm();

    std::vector<shape_t> new_in_shapes;
    for (auto &&in : context.inputs)
        new_in_shapes.emplace_back(in->shape());

    auto new_axis = perm[con.axis()];
    auto new_con = context.graph.emplace<concat>(con.output().type(), new_in_shapes, new_axis);
    auto new_trans = context.graph.emplace<transpose>(con.output().type(), new_con->output().shape(), perm);
    new_trans->input().connect(new_con->output());

    for (size_t i = 0; i < context.inputs.size(); i++)
        new_con->input_at(i).connect(*context.inputs[i]->connection());
    for (auto &&out : dup(con.output().connections()))
        out->connect(new_trans->output());
}

bool transpose_pad_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_pad)
    {
        auto &p = static_cast<pad &>(node);
        if (auto tp = try_get_direct_parent<transpose>(p))
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&p.output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(&p);
            return true;
        }
    }

    return false;
}

void transpose_pad_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_p = static_cast<pad &>(*context.matched_nodes[1]);

    xt::svector<padding> paddings(old_p.paddings().size(), padding::zero());
    for (size_t i = 0; i < paddings.size(); i++)
        paddings[old_tp.perm()[i]] = old_p.paddings()[i];

    auto p = context.graph.emplace<pad>(old_p.output().type(), output.shape(), std::move(paddings), old_p.pad_value());
    auto tp = context.graph.emplace<transpose>(p->output().type(), p->output().shape(), old_tp.perm());
    tp->input().connect(p->output());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool transpose_reduce_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_reduce)
    {
        auto &r = static_cast<reduce &>(node);
        if (auto tp = try_get_direct_parent<transpose>(r))
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&r.output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(&r);
            return true;
        }
    }

    return false;
}

void transpose_reduce_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_r = static_cast<reduce &>(*context.matched_nodes[1]);

    axis_t axis(old_r.axis().size());
    for (size_t i = 0; i < axis.size(); i++)
        axis[i] = old_tp.perm()[old_r.axis()[i]];

    axis_t perm;
    if (old_r.keep_dims())
    {
        perm = old_tp.perm();
    }
    else
    {
        for (auto a : old_tp.perm())
        {
            if (std::find(old_r.axis().begin(), old_r.axis().end(), a) == old_r.axis().end())
            {
                auto idx = std::distance(old_r.axis().begin(), std::lower_bound(old_r.axis().begin(), old_r.axis().end(), a));
                perm.push_back((int32_t)(a - idx));
            }
        }
    }

    auto r = context.graph.emplace<reduce>(old_r.reduce_op(), output.shape(), axis, old_r.init_value(), old_r.keep_dims());
    auto tp = context.graph.emplace<transpose>(r->output().type(), r->output().shape(), perm);
    tp->input().connect(r->output());

    r->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool transpose_unary_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_unary)
    {
        auto &u = static_cast<unary &>(node);
        if (auto tp = try_get_direct_parent<transpose>(u))
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&u.output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(&u);
            return true;
        }
    }

    return false;
}

void transpose_unary_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_u = static_cast<unary &>(*context.matched_nodes[1]);

    auto u = context.graph.emplace<unary>(old_u.unary_op(), output.shape());
    auto tp = context.graph.emplace<transpose>(u->output().type(), u->output().shape(), old_tp.perm());
    tp->input().connect(u->output());

    u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(tp->output());
}
