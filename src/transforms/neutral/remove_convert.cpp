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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/remove_convert.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_converts::on_try_match(node &node, transform_context &context)
{
    if (auto old_convert = node_cast<convert>(node))
    {
        if (try_get_direct_parent<convert>(node))
        {
            context.inputs.emplace_back(&try_get_direct_parent<convert>(node)->input());
            context.outputs.emplace_back(&old_convert->output());

            context.matched_nodes.emplace_back(try_get_direct_parent<convert>(node));
            context.matched_nodes.emplace_back(old_convert);

            return true;
        }
    }

    return false;
}

void fold_converts::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto new_convert = context.graph.emplace<convert>(context.inputs[0]->type(), context.inputs[0]->shape(), context.outputs[0]->type());
    new_convert->name(context.matched_nodes[1]->name());
    new_convert->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(new_convert->output());
}

bool remove_convert::on_try_match(node &node, transform_context &context)
{
    if (auto old_convert = node_cast<convert>(node))
    {
        if (old_convert->input().type() == old_convert->output().type())
        {
            context.inputs.emplace_back(&old_convert->input());
            context.outputs.emplace_back(&old_convert->output());

            context.matched_nodes.emplace_back(old_convert);

            return true;
        }
    }

    return false;
}

void remove_convert::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool remove_concat_convert::on_try_match(node &node, transform_context &context)
{
    if (auto cct = node_cast<concat>(node))
    {
        if (cct->output().type() != quant_type_)
        {
            for (auto &it : cct->inputs())
                context.inputs.emplace_back(it);

            context.outputs.emplace_back(&cct->output());
            context.matched_nodes.emplace_back(cct);
            return true;
        }
    }

    return false;
}

void remove_concat_convert::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto &old_concat = static_cast<concat &>(*context.matched_nodes[0]);

    std::vector<shape_t> in_shape;
    for (size_t it = 0; it < context.inputs.size(); ++it)
    {
        in_shape.emplace_back(context.inputs[it]->shape());
    }
    auto new_concat = context.graph.emplace<concat>(quant_type_, in_shape, old_concat.axis());
    new_concat->name(old_concat.name());
    for (size_t i = 0; i < context.inputs.size(); ++i)
    {
        auto &output = *context.inputs[i]->connection();
        auto in_cvt = context.graph.emplace<convert>(old_concat.output().type(), output.owner().output_at(0).shape(), quant_type_);
        in_cvt->name(new_concat->name() + "/in_convert_" + char(i));
        in_cvt->input().connect(output);
        new_concat->input_at(i).connect(in_cvt->output());
    }
    auto out_cvt = context.graph.emplace<convert>(quant_type_, new_concat->output().shape(), old_concat.output().type());
    out_cvt->name(new_concat->name() + "/out_convert");

    out_cvt->input().connect(new_concat->output());
    for (auto &in : dup(inputs))
        in->connect(out_cvt->output());
}