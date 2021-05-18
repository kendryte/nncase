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
#include <nncase/transforms/neutral/add_quant_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool add_input_dequantize_transform::on_try_match(node &node, transform_context &context)
{
    if (auto in_node = node_cast<input_node>(node))
    {
        if (in_node->output().type() == dt_float32)
        {
            context.outputs.emplace_back(&in_node->output());
            context.matched_nodes.emplace_back(in_node);
            return true;
        }
    }
    return false;
}

void add_input_dequantize_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto old_in = node_cast<input_node>(*context.matched_nodes[0]);

    auto &quantizer = *context.quantizer;
    auto params = quantizer.get_quant_param(quantizer.get(old_in->output()), 8);
    auto new_in_node = context.graph.emplace<input_node>(quant_type_, old_in->output().shape());
    auto deq = context.graph.emplace<dequantize>(new_in_node->output().type(), new_in_node->output().shape(), dt_float32, params);
    deq->input().connect(new_in_node->output());

    new_in_node->name(old_in->name());
    deq->name(new_in_node->name() + "/deq");

    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool add_output_quantize_transform::on_try_match(node &node, transform_context &context)
{
    if (auto out = node_cast<output_node>(node))
    {
        if (out->input().type() == dt_float32)
        {
            context.inputs.emplace_back(&out->input());
            context.matched_nodes.emplace_back(out);
            return true;
        }
    }
    return false;
}

void add_output_quantize_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto old_out = node_cast<output_node>(*context.matched_nodes[0]);
    auto &quantizer = *context.quantizer;
    auto params = quantizer.get_quant_param(quantizer.get(output.owner().output_at(0)), 8);

    auto q = context.graph.emplace<quantize>(dt_float32, output.shape(), quant_type_, params);
    auto new_out_node = context.graph.emplace<output_node>(q->output().type(), q->output().shape());
    old_out->input().clear_connection();

    new_out_node->name(old_out->name());
    q->name(new_out_node->name() + "/q");

    q->input().connect(output);
    new_out_node->input().connect(q->output());
}