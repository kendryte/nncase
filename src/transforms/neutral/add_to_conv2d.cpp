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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/add_to_conv2d.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool add_to_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    binary *add = nullptr;
    if ((add = node_cast<binary>(node))
        && (add->binary_op() == binary_add)
        && (add->input_a().shape().size() == 4)
        && (add->input_a().shape() == add->input_b().shape())
        && (add->input_a().type() == dt_float32)
        && (add->input_b().type() == dt_float32)
        )
    {
        context.inputs.emplace_back(&add->input_a());
        context.inputs.emplace_back(&add->input_b());
        context.outputs.emplace_back(&add->output());

        context.matched_nodes.emplace_back(add);
        return true;
    }

    return false;
}

void add_to_conv2d_transform::process(transform_context &context)
{
    auto &output_a = *context.inputs[0]->connection();
    auto &output_b = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_add = static_cast<binary &>(*context.matched_nodes[0]);

    auto channels = old_add.input_a().shape()[1];
    std::vector<float> weights(channels * 2 * channels);
    for (size_t i = 0; i < channels; i++)
    {
        weights[2 * channels * i + i] = 1.f;
        weights[2 * channels * i + i + channels] = 1.f;
    }

    shape_t in_shapes[] { output_a.shape(), output_b.shape() };
    auto c = context.graph.emplace<concat>(dt_float32, in_shapes, 1);
    c->name(old_add.name() + "/concat");

    auto w_const = context.graph.emplace<constant>(dt_float32, shape_t { channels, 2 * channels, 1, 1 }, weights);
    w_const->name(old_add.name() + "/weights");
    auto conv = context.graph.emplace<conv2d>(c->output().shape(), w_const->output().shape(), 1, padding::zero(),
        padding::zero(), 1, 1, 1, 1, old_add.fused_activation());
    conv->name(old_add.name());

    std::vector<float> bias(channels);
    auto b_const = context.graph.emplace<constant>(dt_float32, shape_t { channels }, bias);
    b_const->name(old_add.name() + "/bias");

    c->input_at(0).connect(output_a);
    c->input_at(1).connect(output_b);
    conv->input().connect(c->output());
    conv->weights().connect(w_const->output());
    conv->bias().connect(b_const->output());

    for (auto &in : dup(inputs))
        in->connect(conv->output());
}