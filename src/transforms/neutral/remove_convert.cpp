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
        datatype_t concat_type = dt_float32;
        if (auto cvt_0 = try_get_direct_parent<convert>(*cct, 0))
        {
            concat_type = cvt_0->input().type();
            context.inputs.emplace_back(&cvt_0->input());
        }
        else
        {
            return false;
        }
        for (size_t i = 1; i < cct->inputs().size(); i++)
        {
            if (auto cvt = try_get_direct_parent<convert>(*cct, i))
            {
                if (cvt->input().type() == concat_type)
                {
                    context.inputs.emplace_back(&cvt->input());
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        if (auto cvt_back = try_get_direct_child<convert>(*cct))
        {
            if (cvt_back->output().type() == concat_type)
            {
                context.outputs.emplace_back(&cvt_back->output());
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        context.matched_nodes.emplace_back(cct);
        return true;
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
    auto new_concat = context.graph.emplace<concat>(context.inputs[0]->connection()->type(), in_shape, old_concat.axis());
    new_concat->name(old_concat.name());
    for (size_t i = 0; i < context.inputs.size(); i++)
    {
        auto &output = *context.inputs[i]->connection();
        new_concat->input_at(i).connect(output);
    }

    for (auto &in : dup(inputs))
        in->connect(new_concat->output());
}