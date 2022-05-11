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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_matmul_add.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_matmul_add_transform::on_try_match(node &node, transform_context &context)
{
    matmul *mm = nullptr;
    binary *add = nullptr;
    bitcast *bc = nullptr;
    constant *bias_constant = nullptr;
    constant *add_constant = nullptr;
    if ((add = node_cast<binary>(node))
        && (add->binary_op() == binary_add)
        && ((((mm = try_get_direct_parent<matmul>(*add, 0)) && (add_constant = try_get_direct_parent<constant>(*add, 1))) || ((mm = try_get_direct_parent<matmul>(*add, 1)) && (add_constant = try_get_direct_parent<constant>(*add, 0)))) || ((bc = try_get_direct_parent<bitcast>(*add, 0)) && (mm = try_get_direct_parent<matmul>(*bc)) && (add_constant = try_get_direct_parent<constant>(*add, 1))) || ((bc = try_get_direct_parent<bitcast>(*add, 1)) && (mm = try_get_direct_parent<matmul>(*bc)) && (add_constant = try_get_direct_parent<constant>(*add, 0))))
        && (mm->fused_activation() == value_range<float>::full())
        && (bias_constant = node_cast<constant>(mm->bias().connection()->owner()))
        && (bias_constant->data().size() == add_constant->data().size()))
    {
        context.inputs.emplace_back(&mm->input_a());
        context.inputs.emplace_back(&mm->input_b());
        context.outputs.emplace_back(&add->output());

        context.matched_nodes.emplace_back(mm);
        context.matched_nodes.emplace_back(bias_constant);
        context.matched_nodes.emplace_back(add);
        context.matched_nodes.emplace_back(add_constant);
        return true;
    }

    return false;
}

void fold_matmul_add_transform::process(transform_context &context)
{
    auto old_mm = node_cast<matmul>(*context.matched_nodes[0]);
    auto old_bias = node_cast<constant>(*context.matched_nodes[1]);
    const float *p = reinterpret_cast<const float *>(old_bias->data().data());

    auto add = node_cast<binary>(*context.matched_nodes[2]);
    auto add_bias = node_cast<constant>(*context.matched_nodes[3]);
    const float *q = reinterpret_cast<const float *>(add_bias->data().data());

    // update bias
    size_t channels = old_mm->bias().connection()->shape().back();
    std::vector<float> bias(channels, 0.f);
    for (size_t i = 0; i < channels; i++)
    {
        bias[i] = p[i] + q[i];
    }

    // update act
    auto mm_act = old_mm->fused_activation();
    auto add_act = add->fused_activation();
    mm_act.min = std::max(mm_act.min, add_act.min);
    mm_act.max = std::min(mm_act.max, add_act.max);

    // create new bias
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t { channels }, std::span(bias));
    new_bias->name(old_bias->name());

    // create new matmul
    auto new_mm = context.graph.emplace<matmul>(old_mm->input_a().shape(), old_mm->input_b().shape(), mm_act);
    auto new_bc = context.graph.emplace<bitcast>(new_mm->output().type(), new_mm->output().shape(), add->output().shape());

    new_bc->name(old_mm->name() + "/bitcast");
    new_mm->name(old_mm->name());
    new_mm->input_a().connect(*context.inputs[0]->connection());
    new_mm->input_b().connect(*context.inputs[1]->connection());
    new_mm->bias().connect(new_bias->output());
    new_bc->input().connect(new_mm->output());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_bc->output());
}
