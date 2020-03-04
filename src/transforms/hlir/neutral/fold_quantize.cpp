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
#include <hlir/ops/dequantize.h>
#include <hlir/ops/quantize.h>
#include <hlir/transforms/neutral/fold_quantize.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool fold_quantize_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_quantize)
    {
        auto &q = static_cast<quantize &>(node);
        for (auto &&conn : q.output().connections())
        {
            if (conn->owner().runtime_opcode() == op_dequantize)
            {
                auto &deq = static_cast<dequantize &>(conn->owner());

                if (almost_equal(q.quant_param(), deq.quant_param()))
                {
                    context.inputs.emplace_back(&q.input());
                    context.outputs.emplace_back(&deq.output());

                    context.matched_nodes.emplace_back(&q);
                    context.matched_nodes.emplace_back(&deq);
                    return true;
                }
            }
        }
    }
    else if (node.runtime_opcode() == op_dequantize)
    {
        auto &deq = static_cast<dequantize &>(node);
        for (auto &&conn : deq.output().connections())
        {
            if (conn->owner().runtime_opcode() == op_quantize)
            {
                auto &q = static_cast<quantize &>(conn->owner());

                if (almost_equal(q.quant_param(), deq.quant_param()))
                {
                    context.inputs.emplace_back(&deq.input());
                    context.outputs.emplace_back(&q.output());

                    context.matched_nodes.emplace_back(&deq);
                    context.matched_nodes.emplace_back(&q);
                    return true;
                }
            }
        }
    }

    return false;
}

void fold_quantize_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_input_quantize_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_input_node)
    {
        auto &in = static_cast<input_node &>(node);
        if (in.output().type() == dt_float32)
        {
            context.outputs.emplace_back(&in.output());

            context.matched_nodes.emplace_back(&in);
            return true;
        }
    }

    return false;
}

void fold_input_quantize_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto &old_in = static_cast<input_node &>(*context.matched_nodes[0]);

    auto input = context.graph.emplace<input_node>(dt_uint8, old_in.output().shape());
    input->name(old_in.name());
    auto deq = context.graph.emplace<dequantize>(input->output().shape(), quantizer_.get_quant_param(quantizer_.get(old_in.output()), 8));
    deq->name(old_in.name() + "/dequantize");
    deq->input().connect(input->output());
    link(old_in.output(), deq->output(), &quantizer_);

    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
