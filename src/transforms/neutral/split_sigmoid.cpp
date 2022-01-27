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
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/split_sigmoid.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool split_sigmoid_transform::on_try_match(node &node, transform_context &context)
{
    if (auto s = node_cast<sigmoid>(node))
    {
        context.inputs.emplace_back(&s->input());
        context.outputs.emplace_back(&s->output());

        context.matched_nodes.emplace_back(s);
        return true;
    }

    return false;
}

void split_sigmoid_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &s = static_cast<sigmoid &>(*context.matched_nodes[0]);

    auto one = context.graph.emplace<constant>(1.f);
    one->name(s.name() + ".one(Sigmoid)");

    auto neg = context.graph.emplace<unary>(unary_neg, s.input().shape());
    neg->name(s.name() + ".neg(Sigmoid)");

    auto exp = context.graph.emplace<unary>(unary_exp, neg->output().shape());
    exp->name(s.name() + ".exp(Sigmoid)");

    auto add = context.graph.emplace<binary>(binary_add, output.type(), one->output().shape(), exp->output().shape(), value_range<float>::nonnegative());
    add->name(s.name() + ".add(Sigmoid)");

    auto div = context.graph.emplace<binary>(binary_div, output.type(), one->output().shape(), add->output().shape(), value_range<float>::nonnegative());
    div->name(s.name() + ".div(Sigmoid)");

    exp->input().connect(neg->output());
    add->input_a().connect(one->output());
    add->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(add->output());

    neg->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(div->output());
}
