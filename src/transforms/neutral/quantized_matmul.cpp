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
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/matmul.h>
#include <ir/ops/quantize.h>
#include <ir/visitor.h>
#include <transforms/neutral/quantized_matmul.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

namespace
{
auto quantize_bn_act(quantizer &quantizer, matmul &mm, float sa, const quant_param_t &yq_p)
{
    xt::xtensor<int32_t, 1> q_bias(mm.bias().shape());
    auto &bias = mm.bias();
    auto so = yq_p.scale / sa;
    auto bn_mul = quantizer.get_fixed_mul(so, 32, 32, true);
    assert(bn_mul.shift > 0);

    for (size_t i = 0; i < bias.size(); i++)
        q_bias[i] = (int32_t)std::round(bias[i] * sa);

    return std::make_tuple(std::move(q_bias), bn_mul);
}
}

bool quantized_matmul_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_matmul)
    {
        auto &mm = static_cast<matmul &>(node);
        if (auto q1 = try_get_direct_parent<fake_quantize>(mm, 0))
        {
            if (auto q2 = try_get_direct_parent<fake_quantize>(mm, 1))
            {
                if (auto deq = try_get_direct_child<fake_dequantize>(mm))
                {
                    context.inputs.emplace_back(&q1->input());
                    context.inputs.emplace_back(&q2->input());
                    context.outputs.emplace_back(&deq->output());

                    context.matched_nodes.emplace_back(q1);
                    context.matched_nodes.emplace_back(q2);
                    context.matched_nodes.emplace_back(&mm);
                    context.matched_nodes.emplace_back(deq);
                    return true;
                }
            }
        }
    }

    return false;
}

void quantized_matmul_transform::process(transform_context &context)
{
    auto &output1 = *context.inputs[0]->connection();
    auto &output2 = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_q1 = static_cast<fake_quantize &>(*context.matched_nodes[0]);
    auto &old_q2 = static_cast<fake_quantize &>(*context.matched_nodes[1]);
    auto &old_mm = static_cast<matmul &>(*context.matched_nodes[2]);
    auto &old_deq = static_cast<fake_dequantize &>(*context.matched_nodes[3]);

    auto i1q_p = quantizer_.get_quant_param(quantizer_.get(old_q1.output()), 8);
    auto i2q_p = quantizer_.get_quant_param(quantizer_.get(old_q2.output()), 8);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_deq.output()), 8);
    auto sa = i1q_p.scale * i2q_p.scale;
    auto [q_bias, act] = quantize_bn_act(quantizer_, old_mm, sa, yq_p);

    auto q1 = context.graph.emplace<quantize>(output1.shape(), i1q_p);
    auto q2 = context.graph.emplace<quantize>(output2.shape(), i2q_p);
    auto mm = context.graph.emplace<quantized_matmul>(q1->output().shape(), q2->output().shape(), std::move(q_bias),
        -i1q_p.zero_point, -i2q_p.zero_point, act.rounded_mul(), act.shift, yq_p.zero_point);
    auto deq = context.graph.emplace<dequantize>(mm->output().shape(), yq_p);
    mm->input_a().connect(q1->output());
    mm->input_b().connect(q2->output());
    deq->input().connect(mm->output());

    q1->input().connect(output1);
    q2->input().connect(output2);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
