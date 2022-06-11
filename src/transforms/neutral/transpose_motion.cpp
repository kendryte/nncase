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
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool transpose_binary_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto bin = node_cast<binary>(node))
    {
        transpose *tp1, *tp2;
        if ((tp1 = node_cast<transpose>(bin->input_a().connection()->owner()))
            && (tp2 = node_cast<transpose>(bin->input_b().connection()->owner())))
        {
            auto output_a_size = bin->input_a().connection()->connections().size();
            auto output_b_size = bin->input_b().connection()->connections().size();
            auto size = tp1 == tp2 ? output_a_size : output_a_size + output_b_size;
            if ((size == 2) && (tp1->perm() == tp2->perm()))
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

    auto bin = context.graph.emplace<binary>(old_bin.binary_op(), output_a.type(), output_a.shape(), output_b.shape(), old_bin.fused_activation());
    bin->attributes(old_bin.attributes());
    bin->name(old_bin.name());
    auto tp = context.graph.emplace<transpose>(bin->output().type(), bin->output().shape(), old_tp.perm());
    tp->name(old_tp.name());
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
    con->name(old_con.name());

    binary *bin;
    if (old_bin.input_a().connection()->owner().runtime_opcode() == op_constant)
    {
        bin = context.graph.emplace<binary>(old_bin.binary_op(), output.type(), con->output().shape(), output.shape(), old_bin.fused_activation());
        bin->attributes(old_bin.attributes());
        bin->name(old_bin.name());
        bin->input_a().connect(con->output());
        bin->input_b().connect(output);
    }
    else
    {
        bin = context.graph.emplace<binary>(old_bin.binary_op(), output.type(), output.shape(), con->output().shape(), old_bin.fused_activation());
        bin->attributes(old_bin.attributes());
        bin->name(old_bin.name());
        bin->input_a().connect(output);
        bin->input_b().connect(con->output());
    }

    auto tp = context.graph.emplace<transpose>(bin->output().type(), bin->output().shape(), old_tp.perm());
    tp->name(old_tp.name());
    tp->input().connect(bin->output());

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
        for (auto in : con.inputs())
        {
            if (in->connection()->owner().runtime_opcode() == op_transpose)
            {
                auto &tp = static_cast<transpose &>(in->connection()->owner());

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
    new_con->name(con.name());
    auto new_trans = context.graph.emplace<transpose>(con.output().type(), new_con->output().shape(), perm);
    new_trans->name(static_cast<transpose &>(*context.matched_nodes[1]).name());
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
        if (auto tp = try_get_direct_child<transpose>(p))
        {
            context.inputs.emplace_back(&p.input());
            context.outputs.emplace_back(&tp->output());

            context.matched_nodes.emplace_back(&p);
            context.matched_nodes.emplace_back(tp);
            return true;
        }
    }

    return false;
}

void transpose_pad_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_p = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[1]);

    // transpose
    auto tp = context.graph.emplace<transpose>(output.type(), output.shape(), old_tp.perm());
    tp->name(old_tp.name());
    tp->input().connect(output);

    // pad
    xt::svector<padding> paddings(old_p.paddings().size(), padding::zero());
    axis_t new_perm { 0, 3, 1, 2 };
    for (size_t i = 0; i < paddings.size(); i++)
        paddings[i] = old_p.paddings()[old_tp.perm()[i]];

    auto p = context.graph.emplace<pad>(tp->output().type(), tp->output().shape(), std::move(paddings), old_p.pad_mode(), old_p.pad_value());
    p->name(old_p.name());
    p->input().connect(tp->output());

    for (auto &in : dup(inputs))
        in->connect(p->output());
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

    axis_t axes(old_r.axis().size());
    for (size_t i = 0; i < axes.size(); i++)
        axes[i] = old_tp.perm()[old_r.axis()[i]];

    axis_t perm = old_tp.perm();
    if (!old_r.keep_dims())
    {
        auto sorted_axes = axes;
        std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int>());
        for (auto axis : sorted_axes)
        {
            auto it = std::find(perm.begin(), perm.end(), axis);
            perm.erase(it);
            std::for_each(perm.begin(), perm.end(), [=](auto &val) {
                val = val > axis ? val - 1 : val;
            });
        }
    }

    auto r = context.graph.emplace<reduce>(old_r.reduce_op(), output.type(), output.shape(), axes, old_r.init_value(), old_r.keep_dims());
    r->attributes(old_r.attributes());
    r->name(old_r.name());
    auto tp = context.graph.emplace<transpose>(r->output().type(), r->output().shape(), perm);
    tp->name(old_tp.name());
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
    u->attributes(old_u.attributes());
    u->name(old_u.name());
    auto tp = context.graph.emplace<transpose>(u->output().type(), u->output().shape(), old_tp.perm());
    tp->name(old_tp.name());
    tp->input().connect(u->output());

    u->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool transpose_clamp_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto c = node_cast<clamp>(node))
    {
        transpose *tp;
        constant *low, *high;
        if ((tp = node_cast<transpose>(c->input().connection()->owner()))
            && (low = node_cast<constant>(c->input_low().connection()->owner()))
            && low->output().shape().size() == 1
            && (high = node_cast<constant>(c->input_high().connection()->owner()))
            && high->output().shape().size() == 1)
        {
            context.inputs.emplace_back(&tp->input());
            context.outputs.emplace_back(&c->output());

            context.matched_nodes.emplace_back(tp);
            context.matched_nodes.emplace_back(low);
            context.matched_nodes.emplace_back(high);
            context.matched_nodes.emplace_back(c);
            return true;
        }
    }

    return false;
}

void transpose_clamp_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_low = static_cast<constant &>(*context.matched_nodes[1]);
    auto &old_high = static_cast<constant &>(*context.matched_nodes[2]);
    auto &old_c = static_cast<clamp &>(*context.matched_nodes[3]);
    auto &old_perm = old_tp.perm();
    auto expand_dim = old_perm.size() - old_perm[old_perm.size() - 1] - 1;

    auto low_shape = old_low.output().shape();
    if (low_shape[0] != 1)
    {
        for (size_t i = 0; i < expand_dim; i++)
            low_shape.push_back(1);
    }
    auto high_shape = old_high.output().shape();
    if (high_shape[0] != 1)
    {
        for (size_t i = 0; i < expand_dim; i++)
            high_shape.push_back(1);
    }

    auto low = context.graph.emplace<constant>(old_low.output().type(), low_shape, old_low.data());
    low->name(old_low.name());
    auto high = context.graph.emplace<constant>(old_high.output().type(), high_shape, old_high.data());
    high->name(old_high.name());

    auto c = context.graph.emplace<clamp>(output.shape(), low->output().shape(), high->output().shape());
    c->name(old_c.name());
    c->input().connect(output);
    c->input_low().connect(low->output());
    c->input_high().connect(high->output());

    auto tp = context.graph.emplace<transpose>(c->output().type(), c->output().shape(), old_tp.perm());
    tp->name(old_tp.name());
    tp->input().connect(c->output());

    for (auto &in : dup(inputs))
        in->connect(tp->output());
}

bool transpose_sigmoid_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto sigmd = node_cast<sigmoid>(node))
    {
        if (auto tp = try_get_direct_parent<transpose>(*sigmd))
        {
            if (tp->output().connections().size() == 1)
            {
                context.inputs.emplace_back(&tp->input());
                context.outputs.emplace_back(&sigmd->output());

                context.matched_nodes.emplace_back(tp);
                context.matched_nodes.emplace_back(sigmd);
                return true;
            }
            else
            {
                // tp-sigmoid has another branch
                if (auto b = try_get_direct_child<binary>(*sigmd))
                {
                    if (tp == node_cast<transpose>(b->input_a().connection()->owner())
                        || tp == node_cast<transpose>(b->input_b().connection()->owner()))
                    {
                        context.inputs.emplace_back(&tp->input());
                        context.outputs.emplace_back(&b->output());

                        context.matched_nodes.emplace_back(tp);
                        context.matched_nodes.emplace_back(sigmd);
                        context.matched_nodes.emplace_back(b);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void transpose_sigmoid_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[0]);
    auto &old_sigmd = static_cast<sigmoid &>(*context.matched_nodes[1]);

    if (context.matched_nodes.size() == 2)
    {
        auto new_sigmd = context.graph.emplace<sigmoid>(old_tp.input().type(), old_tp.input().shape());
        auto new_tp = context.graph.emplace<transpose>(new_sigmd->output().type(), new_sigmd->output().shape(), old_tp.perm());
        new_sigmd->name(old_sigmd.name());
        new_tp->name(old_tp.name());

        new_sigmd->input().connect(output);
        new_tp->input().connect(new_sigmd->output());

        for (auto &in : dup(inputs))
            in->connect(new_tp->output());
    }
    else
    {
        // tp-sigmoid has another branch
        auto &old_b = static_cast<binary &>(*context.matched_nodes[2]);

        auto new_sigmd = context.graph.emplace<sigmoid>(old_tp.input().type(), old_tp.input().shape());
        auto new_b = context.graph.emplace<binary>(old_b.binary_op(), old_tp.input().type(), old_tp.input().shape(), new_sigmd->output().shape(), old_b.fused_activation());
        new_b->attributes(old_b.attributes());
        auto new_tp = context.graph.emplace<transpose>(new_b->output().type(), new_b->output().shape(), old_tp.perm());
        new_sigmd->name(old_sigmd.name());
        new_b->name(old_b.name());
        new_tp->name(old_tp.name());

        new_sigmd->input().connect(output);
        new_b->input_a().connect(output);
        new_b->input_b().connect(new_sigmd->output());
        new_tp->input().connect(new_b->output());

        for (auto &in : dup(inputs))
            in->connect(new_tp->output());
    }
}
