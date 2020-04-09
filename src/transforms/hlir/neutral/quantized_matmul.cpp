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
#include <hlir/ops/dequantize.h>
#include <hlir/ops/matmul.h>
#include <hlir/ops/quantize.h>
#include <hlir/transforms/neutral/quantized_matmul.h>
#include <hlir/visitor.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

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
    if (auto mm = node_cast<matmul>(node))
    {
        if (mm->input_a().connection()->attributes() & cnctr_attr_need_quantize
            && mm->input_b().connection()->attributes() & cnctr_attr_need_quantize
            && mm->output().attributes() & cnctr_attr_need_quantize
            && mm->attributes() & node_attr_need_quantize)
        {
            context.inputs.emplace_back(&mm->input_a());
            context.inputs.emplace_back(&mm->input_b());
            context.outputs.emplace_back(&mm->output());

            context.matched_nodes.emplace_back(mm);
            return true;
        }
    }

    return false;
}

void quantized_matmul_transform::process(transform_context &context)
{
    auto &output1 = *context.inputs[0]->connection();
    auto &output2 = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_mm = static_cast<matmul &>(*context.matched_nodes[0]);

    auto i1q_p = quantizer_.get_quant_param(quantizer_.get(output1), 8);
    auto i2q_p = quantizer_.get_quant_param(quantizer_.get(output2), 8);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_mm.output()), 8);
    auto sa = i1q_p.scale * i2q_p.scale;
    auto [q_bias, act] = quantize_bn_act(quantizer_, old_mm, sa, yq_p);

    auto q1 = context.graph.emplace<quantize>(output1.shape(), i1q_p);
    q1->name(output1.owner().name() + "/quantize");
    auto q2 = context.graph.emplace<quantize>(output2.shape(), i2q_p);
    q2->name(output2.owner().name() + "/quantize");
    auto mm = context.graph.emplace<quantized_matmul>(q1->output().shape(), q2->output().shape(), std::move(q_bias),
        -i1q_p.zero_point, -i2q_p.zero_point, act.rounded_mul(), act.shift, yq_p.zero_point);
    mm->name(old_mm.name());
    auto deq = context.graph.emplace<dequantize>(mm->output().shape(), yq_p);
    deq->name(old_mm.name() + "/dequantize");
    link(old_mm.output(), deq->output(), &quantizer_);

    mm->input_a().connect(q1->output());
    mm->input_b().connect(q2->output());
    deq->input().connect(mm->output());

    q1->input().connect(output1);
    q2->input().connect(output2);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
