/* Copyright 2019 Canaan Inc.
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
#include <algorithm>
#include <ir/ops/binary.h>
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/quantize.h>
#include <ir/visitor.h>
#include <transforms/neutral/quantized_binary.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool quantized_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (auto q1 = try_get_direct_parent<fake_quantize>(*b, 0))
        {
            if (auto q2 = try_get_direct_parent<fake_quantize>(*b, 1))
            {
                if (auto deq = try_get_direct_child<fake_dequantize>(*b))
                {
                    context.inputs.emplace_back(&q1->input());
                    context.inputs.emplace_back(&q2->input());
                    context.outputs.emplace_back(&deq->output());

                    context.matched_nodes.emplace_back(q1);
                    context.matched_nodes.emplace_back(q2);
                    context.matched_nodes.emplace_back(b);
                    context.matched_nodes.emplace_back(deq);
                    return true;
                }
            }
        }
    }

    return false;
}

void quantized_binary_transform::process(transform_context &context)
{
    auto &output1 = *context.inputs[0]->connection();
    auto &output2 = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_q1 = static_cast<fake_quantize &>(*context.matched_nodes[0]);
    auto &old_q2 = static_cast<fake_quantize &>(*context.matched_nodes[1]);
    auto &old_b = static_cast<binary &>(*context.matched_nodes[2]);
    auto &old_deq = static_cast<fake_dequantize &>(*context.matched_nodes[3]);

    auto i1q_p = quantizer_.get_quant_param(quantizer_.get(old_q1.output()), 8);
    auto i2q_p = quantizer_.get_quant_param(quantizer_.get(old_q2.output()), 8);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_deq.output()), 8);
    float sa;
    fixed_mul fix_a, fix_b;

    switch (old_b.binary_op())
    {
    case binary_add:
    case binary_sub:
    case binary_min:
    case binary_max:
        sa = i1q_p.scale * i2q_p.scale;
        fix_a = quantizer_.get_fixed_mul(i2q_p.scale, 16, 16, true);
        fix_b = quantizer_.get_fixed_mul(i1q_p.scale, 16, 16, true);
        break;
    case binary_mul:
        sa = i1q_p.scale * i2q_p.scale;
        fix_a = { 1.f, 0 };
        fix_b = { 1.f, 0 };
        break;
    case binary_div:
        sa = i1q_p.scale;
        fix_a = quantizer_.get_fixed_mul(i2q_p.scale, 16, 16, true);
        fix_b = { 1.f, 0 };
        break;
    default:
        throw std::runtime_error("Unsupported quantized binary op");
    }

    auto fix_y = quantizer_.get_fixed_mul(yq_p.scale / sa, 32, 32, true);

    auto q1 = context.graph.emplace<quantize>(output1.shape(), i1q_p);
    auto q2 = context.graph.emplace<quantize>(output2.shape(), i2q_p);
    auto b = context.graph.emplace<quantized_binary>(old_b.binary_op(), q1->output().shape(), q2->output().shape(),
        -i1q_p.zero_point, fix_a.rounded_mul(), fix_a.shift, -i2q_p.zero_point, fix_b.rounded_mul(), fix_b.shift,
        fix_y.rounded_mul(), fix_y.shift, yq_p.zero_point);
    auto deq = context.graph.emplace<dequantize>(b->output().shape(), yq_p);
    b->input_a().connect(q1->output());
    b->input_b().connect(q2->output());
    deq->input().connect(b->output());

    q1->input().connect(output1);
    q2->input().connect(output2);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
