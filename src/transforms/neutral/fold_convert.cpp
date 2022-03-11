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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_convert.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_convert_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cvt1 = node_cast<convert>(node))
    {
        if (auto cvt2 = try_get_direct_child<convert>(*cvt1))
        {
            context.inputs.emplace_back(&cvt1->input());
            context.outputs.emplace_back(&cvt2->output());

            context.matched_nodes.emplace_back(cvt1);
            context.matched_nodes.emplace_back(cvt2);
            return true;
        }
    }

    return false;
}

void fold_convert_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &cvt2 = *static_cast<convert *>(context.matched_nodes[1]);

    auto cvt = context.graph.emplace<convert>(output.type(), output.shape(), cvt2.output().type());
    cvt->name(cvt2.name());

    cvt->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(cvt->output());
}

bool fold_nop_convert_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cvt = node_cast<convert>(node))
    {
        if (cvt->input().type() == cvt->output().type())
        {
            context.inputs.emplace_back(&cvt->input());
            context.outputs.emplace_back(&cvt->output());

            context.matched_nodes.emplace_back(cvt);
            return true;
        }
    }

    return false;
}

void fold_nop_convert_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_convert_around_bitcast_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cvt = node_cast<convert>(node))
    {
        if (auto bitc = try_get_direct_parent<bitcast>(*cvt))
        {
            if (auto sec_cvt = try_get_direct_parent<convert>(*bitc))
            {
                context.inputs.emplace_back(&sec_cvt->input());
                context.outputs.emplace_back(&cvt->output());

                context.matched_nodes.emplace_back(bitc);
                return true;
            }
        }
    }

    return false;
}

void fold_convert_around_bitcast_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto mid_node = node_cast<bitcast>(*context.matched_nodes[0]);
    auto new_bitc = context.graph.emplace<bitcast>(output.type(), output.shape(), inputs[0]->type(), inputs[0]->shape());
    new_bitc->name(mid_node->name());

    new_bitc->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(new_bitc->output());
}
