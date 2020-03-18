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
#include <hlir/ops/pad.h>
#include <hlir/ops/reshape.h>
#include <hlir/ops/strided_slice.h>
#include <hlir/transforms/neutral/fold_pad.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

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
            if (p1->pad_value() == p2->pad_value())
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

    auto p = context.graph.emplace<pad>(old_p1.input().type(), output.shape(), paddings, old_p1.pad_value());
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
            if (auto slice = try_get_direct_child<strided_slice>(p))
            {
                if (std::all_of(slice->begin().begin(), slice->begin().end(), [](const int32_t &value) { return value >= 0; })
                    && std::all_of(slice->end().begin(), slice->end().end(), [](const int32_t &value) { return value >= 0; })
                    && slice->new_axis_mask() == 0
                    && slice->shrink_axis_mask() == 0)
                {
                    context.inputs.emplace_back(&p.input());
                    context.outputs.emplace_back(&slice->output());

                    context.matched_nodes.emplace_back(&p);
                    context.matched_nodes.emplace_back(slice);
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
    auto &old_slice = static_cast<strided_slice &>(*context.matched_nodes[1]);

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

    auto p = context.graph.emplace<pad>(old_p.input().type(), output.shape(), paddings, old_p.pad_value());
    p->name(old_p.name());
    auto slice = context.graph.emplace<strided_slice>(p->output().type(), p->output().shape(), begin, end, old_slice.strides(),
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask(), old_slice.shrink_axis_mask());
    slice->name(old_slice.name());
    slice->input().connect(p->output());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(slice->output());
}

bool strided_slice_to_pad_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_strided_slice)
    {
        auto &slice = static_cast<strided_slice &>(node);

        if (slice.strides() == axis_t { 1, 1, 1, 1 })
        {
            context.inputs.emplace_back(&slice.input());
            context.outputs.emplace_back(&slice.output());

            context.matched_nodes.emplace_back(&slice);
            return true;
        }
    }

    return false;
}

void strided_slice_to_pad_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<strided_slice &>(*context.matched_nodes[0]);

    auto begin = old_slice.begin();
    auto end = old_slice.end();

    xt::svector<padding> paddings(output.shape().size(), padding::zero());
    for (size_t i = 0; i < paddings.size(); i++)
        paddings[i] = { -begin[i], end[i] - (int32_t)output.shape()[i] };

    auto p = context.graph.emplace<pad>(output.type(), output.shape(), paddings, (uint8_t)0); // dummy pad value because of cropping
    p->name(old_slice.name());
    auto rshape = context.graph.emplace<reshape>(output.type(), p->output().shape(), old_slice.output().shape());
    rshape->input().connect(p->output());

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rshape->output());
}
