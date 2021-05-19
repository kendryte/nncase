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
#include <nncase/transforms/neutral/fold_io_quant_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_i_quant_transform::on_try_match(node &node, transform_context &context)
{
    quantize *q;
    dequantize *deq;
    if (auto in_node = node_cast<input_node>(node))
    {
        if ((in_node->output().type() == dt_float32)
            && (q = try_get_direct_child<quantize>(*in_node))
            && (deq = try_get_direct_child<dequantize>(*q)))
        {
            context.inputs.emplace_back(&q->input());
            context.outputs.emplace_back(&deq->output());
            return true;
        }
    }
    return false;
}

void fold_i_quant_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_o_quant_transform::on_try_match(node &node, transform_context &context)
{
    quantize *q;
    dequantize *deq;
    if (auto out_node = node_cast<output_node>(node))
    {
        if ((out_node->input().type() == dt_float32)
            && (deq = try_get_direct_parent<dequantize>(*out_node))
            && (q = try_get_direct_parent<quantize>(*deq)))
        {
            context.inputs.emplace_back(&q->input());
            context.outputs.emplace_back(&deq->output());
            return true;
        }
    }
    return false;
}

void fold_o_quant_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
