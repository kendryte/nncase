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
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_quantize.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

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
                    // if ((try_get_direct_parent<space_to_batch>(q) && try_get_direct_child<conv2d>(deq))
                    //     || (try_get_direct_parent<conv2d>(q) && try_get_direct_child<batch_to_space>(deq)))
                    // {
                    return true;
                    // }
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
                if (almost_equal(q.quant_param(), deq.quant_param()) && q.output().type() == deq.input().type())
                {
                    context.inputs.emplace_back(&deq.input());
                    context.outputs.emplace_back(&q.output());

                    context.matched_nodes.emplace_back(&deq);
                    context.matched_nodes.emplace_back(&q);
                    return true;
                }
                else
                {
                    return false;
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

bool fold_quantize_around_bitcast_transform::on_try_match(node &node, transform_context &context)
{
    if (auto q = node_cast<quantize>(node))
    {
        if (auto bitc = try_get_direct_parent<bitcast>(*q))
        {
            if (auto deq = try_get_direct_parent<dequantize>(*bitc))
            {
                if (almost_equal(q->quant_param(), deq->quant_param()))
                {
                    context.inputs.emplace_back(&deq->input());
                    context.outputs.emplace_back(&q->output());

                    context.matched_nodes.emplace_back(bitc);
                    return true;
                }
            }
        }
    }
    return false;
}

void fold_quantize_around_bitcast_transform::process(transform_context &context)
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