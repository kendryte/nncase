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
#include <algorithm>
#include <hlir/ops/binary.h>
#include <hlir/ops/dequantize.h>
#include <hlir/ops/quantize.h>
#include <hlir/transforms/neutral/quantized_binary.h>
#include <hlir/visitor.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool quantized_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        if (b->input_a().connection()->attributes() & cnctr_attr_need_quantize
            && b->input_b().connection()->attributes() & cnctr_attr_need_quantize
            && b->output().attributes() & cnctr_attr_need_quantize)
        {
            context.inputs.emplace_back(&b->input_a());
            context.inputs.emplace_back(&b->input_b());
            context.outputs.emplace_back(&b->output());

            context.matched_nodes.emplace_back(b);
            return true;
        }
    }

    return false;
}

void quantized_binary_transform::process(transform_context &context)
{
    auto &output1 = *context.inputs[0]->connection();
    auto &output2 = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_b = static_cast<binary &>(*context.matched_nodes[0]);

    auto i1q_p = quantizer_.get_quant_param(quantizer_.get(output1), 8);
    auto i2q_p = quantizer_.get_quant_param(quantizer_.get(output2), 8);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_b.output()), 8);
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
    q1->name(output1.owner().name() + "/quantize");
    auto q2 = context.graph.emplace<quantize>(output2.shape(), i2q_p);
    q2->name(output2.owner().name() + "/quantize");
    auto b = context.graph.emplace<quantized_binary>(old_b.binary_op(), q1->output().shape(), q2->output().shape(),
        -i1q_p.zero_point, fix_a.rounded_mul(), fix_a.shift, -i2q_p.zero_point, fix_b.rounded_mul(), fix_b.shift,
        fix_y.rounded_mul(), fix_y.shift, yq_p.zero_point);
    b->name(old_b.name());
    auto deq = context.graph.emplace<dequantize>(b->output().shape(), yq_p);
    deq->name(old_b.name() + "/dequantize");
    link(old_b.output(), deq->output(), &quantizer_);

    b->input_a().connect(q1->output());
    b->input_b().connect(q2->output());
    deq->input().connect(b->output());

    q1->input().connect(output1);
    q2->input().connect(output2);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
