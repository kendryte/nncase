/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/clamp_to_binary.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool clamp_to_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto cl = node_cast<clamp>(node))
    {
        context.inputs.emplace_back(&cl->input());
        context.inputs.emplace_back(&cl->input_low());
        context.inputs.emplace_back(&cl->input_high());
        context.outputs.emplace_back(&cl->output());

        context.matched_nodes.emplace_back(cl);
        return true;
    }

    return false;
}

void clamp_to_binary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &low = *context.inputs[1]->connection();
    auto &high = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto max = context.graph.emplace<binary>(binary_max, output.shape(), low.shape(), value_range<float>::full());
    max->name(context.matched_nodes[0]->name());
    auto min = context.graph.emplace<binary>(binary_min, max->output().shape(), high.shape(), value_range<float>::full());
    min->name(context.matched_nodes[0]->name());

    max->input_a().connect(output);
    max->input_b().connect(low);
    min->input_a().connect(max->output());
    min->input_b().connect(high);
    for (auto &in : dup(inputs))
        in->connect(min->output());
}
