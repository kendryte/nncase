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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/matmul_to_conv2d.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool matmul_to_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto mm = node_cast<matmul>(node))
    {
        context.inputs.emplace_back(&mm->input_a());
        context.inputs.emplace_back(&mm->input_b());
        context.inputs.emplace_back(&mm->bias());
        context.outputs.emplace_back(&mm->output());

        context.matched_nodes.emplace_back(mm);
        return true;
    }

    return false;
}

void matmul_to_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_mm = static_cast<matmul &>(*context.matched_nodes[0]);

    shape_t if_shape {
        output.shape()[0],
        output.shape()[1],
        1,
        1,
    };
    shape_t w_shape {
        weights.shape()[1],
        weights.shape()[0],
        1,
        1,
    };

    auto if_rshape = context.graph.emplace<bitcast>(output.type(), output.shape(), if_shape);
    if_rshape->name(old_mm.name() + "/if_reshape");
    auto w_tp = context.graph.emplace<transpose>(weights.type(), weights.shape(), axis_t { 1, 0 });
    w_tp->name(old_mm.name() + "/w_transpose");
    auto w_rshape = context.graph.emplace<bitcast>(w_tp->output().type(), w_tp->output().shape(), w_shape);
    w_rshape->name(old_mm.name() + "/w_reshape");
    auto conv = context.graph.emplace<conv2d>(if_rshape->output().shape(), w_rshape->output().shape(), 1,
        padding::zero(), padding::zero(), 1, 1, 1, 1, old_mm.fused_activation());
    auto of_rshape = context.graph.emplace<bitcast>(conv->output().type(), conv->output().shape(), old_mm.output().shape());
    of_rshape->name(old_mm.name() + "/of_reshape");
    conv->name(old_mm.name());
    w_rshape->input().connect(w_tp->output());
    conv->input().connect(if_rshape->output());
    conv->weights().connect(w_rshape->output());
    conv->bias().connect(bias);
    of_rshape->input().connect(conv->output());

    if_rshape->input().connect(output);
    w_tp->input().connect(weights);
    for (auto &in : dup(inputs))
        in->connect(of_rshape->output());
}
