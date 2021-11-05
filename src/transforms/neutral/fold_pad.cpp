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
#include <nncase/ir/ir_types.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_pad.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_nop_pad_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_pad)
    {
        auto &p = static_cast<pad &>(node);

        if (std::all_of(p.paddings().begin(), p.paddings().end(), [](const padding &value) { return value == padding::zero(); }))
        {
            context.inputs.emplace_back(&p.input());
            context.outputs.emplace_back(&p.output());

            context.matched_nodes.emplace_back(&p);
            return true;
        }
    }

    return false;
}

void fold_nop_pad_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_pad_pad_transform::on_try_match(node &node, transform_context &context)
{
    if (auto p1 = node_cast<pad>(node))
    {
        if (auto p2 = try_get_direct_child<pad>(*p1))
        {
            if (p1->pad_mode() == p2->pad_mode()
                && (p1->pad_mode() != pad_constant || p1->pad_value() == p2->pad_value()))
            {
                context.inputs.emplace_back(&p1->input());
                context.outputs.emplace_back(&p2->output());

                context.matched_nodes.emplace_back(p1);
                context.matched_nodes.emplace_back(p2);
                return true;
            }
        }
    }

    return false;
}

void fold_pad_pad_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_p1 = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_p2 = static_cast<pad &>(*context.matched_nodes[1]);

    xt::svector<padding> paddings { old_p1.paddings()[0] + old_p2.paddings()[0],
        old_p1.paddings()[1] + old_p2.paddings()[1],
        old_p1.paddings()[2] + old_p2.paddings()[2],
        old_p1.paddings()[3] + old_p2.paddings()[3] };

    auto p = context.graph.emplace<pad>(old_p1.input().type(), output.shape(), paddings, old_p1.pad_mode(), old_p1.pad_value());
    p->name(old_p1.name());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(p->output());
}

bool fold_pad_strided_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_pad)
    {
        auto &p = static_cast<pad &>(node);

        if (std::any_of(p.paddings().begin(), p.paddings().end(), [](const padding &value) { return value.before < 0 || value.after < 0; }))
        {
            if (auto sl = try_get_direct_child<slice>(p))
            {
                if (std::all_of(sl->begin().begin(), sl->begin().end(), [](const int32_t &value) { return value >= 0; })
                    && std::all_of(sl->end().begin(), sl->end().end(), [](const int32_t &value) { return value >= 0; })
                    && sl->new_axis_mask() == 0)
                {
                    context.inputs.emplace_back(&p.input());
                    context.outputs.emplace_back(&sl->output());

                    context.matched_nodes.emplace_back(&p);
                    context.matched_nodes.emplace_back(sl);
                    return true;
                }
            }
        }
    }

    return false;
}

void fold_pad_strided_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_p = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_slice = static_cast<slice &>(*context.matched_nodes[1]);

    auto paddings = old_p.paddings();
    auto begin = old_slice.begin();
    auto end = old_slice.end();

    for (size_t i = 0; i < paddings.size(); i++)
    {
        auto &padding = paddings[i];
        if (padding.before < 0)
        {
            auto pad_before = -padding.before;
            begin[i] += pad_before;
            end[i] += pad_before;
            padding.before = 0;
        }
        if (padding.after < 0)
        {
            padding.after = 0;
        }
    }

    auto p = context.graph.emplace<pad>(old_p.input().type(), output.shape(), paddings, old_p.pad_mode(), old_p.pad_value());
    p->name(old_p.name());
    auto sl = context.graph.emplace<slice>(p->output().type(), p->output().shape(), begin, end, old_slice.strides(),
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask());
    sl->name(old_slice.name());
    sl->input().connect(p->output());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(sl->output());
}

bool strided_slice_to_pad_transform::on_try_match(node &node, transform_context &context)
{
    if (auto sl = node_cast<slice>(node))
    {
        if (sl->strides() == axis_t { 1, 1, 1, 1 })
        {
            context.inputs.emplace_back(&sl->input());
            context.outputs.emplace_back(&sl->output());

            context.matched_nodes.emplace_back(sl);
            return true;
        }
    }

    return false;
}

void strided_slice_to_pad_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<slice &>(*context.matched_nodes[0]);

    auto begin = old_slice.begin();
    auto end = old_slice.end();

    xt::svector<padding> paddings(output.shape().size(), padding::zero());
    for (size_t i = 0; i < paddings.size(); i++)
        paddings[i] = { -begin[i], end[i] - (int32_t)output.shape()[i] };

    auto p = context.graph.emplace<pad>(output.type(), output.shape(), paddings, pad_constant, (uint8_t)0); // dummy pad value because of cropping
    p->name(old_slice.name());
    auto rshape = context.graph.emplace<bitcast>(output.type(), p->output().shape(), old_slice.output().shape());
    rshape->name(old_slice.name() + "_F");
    rshape->input().connect(p->output());

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rshape->output());
}
bool pad_to_slice_transform::on_try_match(node &node, transform_context &context)
{
    if (auto p = node_cast<pad>(node))
    {
        bool flag = true;
        for (auto &padding : p->paddings())
        {
            if (padding.before > 0 || padding.after > 0)
                flag = false;
        }
        if (flag)
        {
            context.inputs.emplace_back(&p->input());
            context.outputs.emplace_back(&p->output());

            context.matched_nodes.emplace_back(p);
            return true;
        }
    }

    return false;
}

void pad_to_slice_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_pad = static_cast<pad &>(*context.matched_nodes[0]);

    auto paddings = old_pad.paddings();
    axis_t begin(output.shape().size(), 0);
    axis_t end(output.shape().size(), 0);

    for (size_t i = 0; i < paddings.size(); i++)
    {
        begin[i] = -paddings[i].before;
        end[i] = paddings[i].after + output.shape()[i];
    }

    auto sl = context.graph.emplace<slice>(output.type(), output.shape(), begin, end);
    sl->name(old_pad.name());

    sl->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(sl->output());
}

bool pad_to_maxpool_transform::on_try_match(node &node, transform_context &context)
{
    if (auto pd = node_cast<pad>(node))
    {
        if (auto mxp = try_get_direct_child<reduce_window2d>(*pd))
        {
            if (mxp->reduce_op() == reduce_op_t::reduce_max)
            {
                if (mxp->padding_h().before == 0 && mxp->padding_h().after == 0 && mxp->padding_w().before == 0 && mxp->padding_w().after == 0)
                {
                    context.inputs.emplace_back(&pd->input());
                    context.outputs.emplace_back(&mxp->output());

                    context.matched_nodes.emplace_back(pd);
                    context.matched_nodes.emplace_back(mxp);
                    return true;
                }
            }
        }
    }

    return false;
}

void pad_to_maxpool_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &pd = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_mxp = static_cast<reduce_window2d &>(*context.matched_nodes[1]);

    auto pad_h = pd.paddings()[2];
    auto pad_W = pd.paddings()[3];
    auto new_mxp = context.graph.emplace<ir::reduce_window2d>(reduce_max, pd.input().shape(), old_mxp.init_value(),
        old_mxp.filter_h(), old_mxp.filter_w(), pad_h, pad_W, old_mxp.stride_h(), old_mxp.stride_w(),
        old_mxp.dilation_h(), old_mxp.dilation_w(), old_mxp.fused_activation(), old_mxp.ceil_mode(), old_mxp.count_include_pad(), old_mxp.padding_h_w_after(), old_mxp.strict_inside_input(), pd.pad_value());
    new_mxp->name(old_mxp.name());

    new_mxp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(new_mxp->output());
}