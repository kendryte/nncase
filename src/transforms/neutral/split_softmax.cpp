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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/softmax.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/split_softmax.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool split_softmax_transform::on_try_match(node &node, transform_context &context)
{
    if (auto sm = node_cast<softmax>(node))
    {
        context.inputs.emplace_back(&sm->input());
        context.outputs.emplace_back(&sm->output());

        context.matched_nodes.emplace_back(sm);
        return true;
    }

    return false;
}

void split_softmax_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &sm = static_cast<softmax &>(*context.matched_nodes[0]);

    auto input_type = output.type();
    auto input_shape = output.shape();
    axis_t axes { sm.axis() };
    auto rmax = context.graph.emplace<reduce>(reduce_max, input_type, input_shape, axes, std::numeric_limits<float>::lowest(), true);
    rmax->attributes(rmax->attributes() | node_attributes::node_attr_skip_quantize);
    rmax->name(sm.name() + ".rmax");

    auto sub = context.graph.emplace<binary>(binary_sub, input_type, input_shape, rmax->output().shape(), value_range<float>::full());
    sub->attributes(sub->attributes() | node_attributes::node_attr_skip_quantize);
    sub->name(sm.name() + ".sub");

    auto beta = context.graph.emplace<constant>(sm.beta());
    beta->name(sm.name() + ".beta");

    auto mul = context.graph.emplace<binary>(binary_mul, input_type, sub->output().shape(), beta->output().shape(), value_range<float>::full());
    mul->attributes(mul->attributes() | node_attributes::node_attr_skip_quantize);
    mul->name(sm.name() + ".mul");

    auto exp = context.graph.emplace<unary>(unary_exp, sub->output().shape());
    exp->attributes(exp->attributes() | node_attributes::node_attr_skip_quantize);
    exp->name(sm.name() + ".exp");

    auto rsum = context.graph.emplace<reduce>(reduce_sum, input_type, exp->output().shape(), axes, 0.f, true);
    rsum->attributes(rsum->attributes() | node_attributes::node_attr_skip_quantize);
    rsum->name(sm.name() + ".rsum");

    auto div = context.graph.emplace<binary>(binary_div, input_type, exp->output().shape(), rsum->output().shape(), value_range<float>::full());
    div->attributes(div->attributes() | node_attributes::node_attr_skip_quantize);
    div->name(sm.name() + ".div");

    rmax->input().connect(output);
    sub->input_a().connect(output);
    sub->input_b().connect(rmax->output());
    mul->input_a().connect(sub->output());
    mul->input_b().connect(beta->output());
    exp->input().connect(mul->output());
    rsum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(rsum->output());

    for (auto &in : dup(inputs))
        in->connect(div->output());
}
