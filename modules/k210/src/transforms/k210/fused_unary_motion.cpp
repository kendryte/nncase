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
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/transforms/k210/fused_unary_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

bool slice_fused_unary_motion_transform::on_try_match(
    node &node, transform_context &context) {
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d) {
        slice *s;
        fused_unary *fu;
        if ((s = try_get_direct_child<slice>(node)) &&
            (fu = try_get_direct_child<fused_unary>(*s))) {
            context.inputs.emplace_back(&s->input());
            context.outputs.emplace_back(&fu->output());

            context.matched_nodes.emplace_back(s);
            context.matched_nodes.emplace_back(fu);
            return true;
        }
    }

    return false;
}

void slice_fused_unary_motion_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_slice = static_cast<slice &>(*context.matched_nodes[0]);
    auto &old_fu = static_cast<fused_unary &>(*context.matched_nodes[1]);

    auto fu =
        context.graph.emplace<fused_unary>(old_fu.subgraph(), output.shape());
    fu->name(old_fu.name());
    fu->input().connect(output);
    old_slice.input().connect(fu->output());

    for (auto &in : dup(inputs))
        in->connect(old_slice.output());
}

bool pad_fused_unary_motion_transform::on_try_match(
    node &node, transform_context &context) {
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d) {
        pad *p;
        fused_unary *fu;
        if ((p = try_get_direct_child<pad>(node)) &&
            (fu = try_get_direct_child<fused_unary>(*p))) {
            context.inputs.emplace_back(&p->input());
            context.outputs.emplace_back(&fu->output());

            context.matched_nodes.emplace_back(p);
            context.matched_nodes.emplace_back(fu);
            return true;
        }
    }

    return false;
}

void pad_fused_unary_motion_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_pad = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_fu = static_cast<fused_unary &>(*context.matched_nodes[1]);

    auto fu =
        context.graph.emplace<fused_unary>(old_fu.subgraph(), output.shape());
    fu->name(old_fu.name());
    fu->input().connect(output);
    old_pad.input().connect(fu->output());

    for (auto &in : dup(inputs))
        in->connect(old_pad.output());
}
