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
#include <ir/ops/pad.h>
#include <ir/ops/strided_slice.h>
#include <ir/visitor.h>
#include <transforms/neutral/fold_pad.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

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
    auto slice = context.graph.emplace<strided_slice>(p->output().type(), p->output().shape(), begin, end, old_slice.strides(),
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask(), old_slice.shrink_axis_mask());
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

    auto p = context.graph.emplace<pad>(output.type(), output.shape(), paddings, 0); // dummy pad value because of cropping

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(p->output());
}
