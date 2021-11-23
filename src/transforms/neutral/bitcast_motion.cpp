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
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/bitcast_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool bitcast_clamp_motion_transform::on_try_match(node &node, transform_context &context)
{
    constant *low, *high;
    if (auto bc = node_cast<bitcast>(node))
    {
        if (auto cl = try_get_direct_child<clamp>(*bc))
        {
            if ((low = node_cast<constant>(cl->input_low().connection()->owner()))
                && low->output().shape().size() == 1
                && (high = node_cast<constant>(cl->input_high().connection()->owner()))
                && high->output().shape().size() == 1)
            {
                context.inputs.emplace_back(&bc->input());
                context.outputs.emplace_back(&cl->output());

                context.matched_nodes.emplace_back(bc);
                context.matched_nodes.emplace_back(cl);
                context.matched_nodes.emplace_back(low);
                context.matched_nodes.emplace_back(high);
                return true;
            }
        }
    }

    return false;
}

void bitcast_clamp_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_bc = static_cast<bitcast &>(*context.matched_nodes[0]);
    auto &old_cl = static_cast<clamp &>(*context.matched_nodes[1]);
    auto &old_low = static_cast<constant &>(*context.matched_nodes[2]);
    auto &old_high = static_cast<constant &>(*context.matched_nodes[3]);

    auto new_cl = context.graph.emplace<clamp>(output.shape(), old_cl.input_low().shape(), old_cl.input_high().shape());
    new_cl->name(old_cl.name());
    auto new_bc = context.graph.emplace<bitcast>(old_bc.input().type(), new_cl->output().shape(), old_bc.output().shape());
    new_bc->name(old_bc.name());
    auto new_low = context.graph.emplace<constant>(old_low.output().type(), old_cl.input_low().shape(), old_low.data());
    new_low->name(old_low.name());
    auto new_high = context.graph.emplace<constant>(old_high.output().type(), old_cl.input_high().shape(), old_high.data());
    new_high->name(old_high.name());

    new_cl->input().connect(output);
    new_cl->input_low().connect(new_low->output());
    new_cl->input_high().connect(new_high->output());
    new_bc->input().connect(new_cl->output());

    for (auto &in : dup(inputs))
        in->connect(new_bc->output());
}
