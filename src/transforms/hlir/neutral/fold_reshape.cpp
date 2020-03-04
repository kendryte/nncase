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
#include <hlir/ops/reshape.h>
#include <hlir/transforms/neutral/fold_reshape.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool fold_reshape_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_reshape)
    {
        auto &rp1 = static_cast<reshape &>(node);
        if (auto rp2 = try_get_direct_child<reshape>(rp1))
        {
            context.inputs.emplace_back(&rp1.input());
            context.outputs.emplace_back(&rp2->output());

            context.matched_nodes.emplace_back(&rp1);
            context.matched_nodes.emplace_back(rp2);
            return true;
        }
    }

    return false;
}

void fold_reshape_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &rp2 = *static_cast<reshape *>(context.matched_nodes[1]);

    auto rp = context.graph.emplace<reshape>(output.type(), output.shape(), rp2.output().shape());
    rp->name(rp2.name());

    rp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rp->output());
}

bool fold_nop_reshape_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_reshape)
    {
        auto &rp = static_cast<reshape &>(node);

        if (rp.input().shape() == rp.output().shape())
        {
            context.inputs.emplace_back(&rp.input());
            context.outputs.emplace_back(&rp.output());

            context.matched_nodes.emplace_back(&rp);
            return true;
        }
    }

    return false;
}

void fold_nop_reshape_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
