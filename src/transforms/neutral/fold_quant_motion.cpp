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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_quant_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_quant_input_transform::on_try_match(node &node, transform_context &context)
{
    if (auto in_node = node_cast<input_node>(node))
    {
        if (auto old_quant = try_get_direct_child<quantize>(*in_node))
        {
            context.outputs.emplace_back(&old_quant->output());
            context.matched_nodes.emplace_back(old_quant);
            return true;
        }
    }
    return false;
}

void fold_quant_input_transform::process(transform_context &context)
{
    auto old_quant = node_cast<quantize>(*context.matched_nodes[0]);
    auto inputs = context.outputs[0]->connections();

    auto new_in_node = context.graph.emplace<input_node>(old_quant->output().type(), old_quant->output().shape());

    for (auto &in : dup(inputs))
        in->connect(new_in_node->output());
}

bool fold_dequant_output_transform::on_try_match(node &node, transform_context &context)
{
    if (auto old_dequant = node_cast<dequantize>(node))
    {
        if (try_get_direct_child<output_node>(*old_dequant))
        {
            context.inputs.emplace_back(&old_dequant->input());
            context.matched_nodes.emplace_back(old_dequant);
            return true;
        }
    }
    return false;
}

void fold_dequant_output_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto *old_dequant = node_cast<dequantize>(*context.matched_nodes[0]);
    auto new_out_node = context.graph.emplace<output_node>(old_dequant->input().type(), old_dequant->input().shape());
    old_dequant->output().clear_connections();

    new_out_node->input().connect(output);
}