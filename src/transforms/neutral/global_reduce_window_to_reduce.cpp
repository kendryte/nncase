/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/global_reduce_window_to_reduce.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool global_reduce_window_to_reduce_transform::on_try_match(node &node, transform_context &context)
{
    if (auto rw = node_cast<reduce_window2d>(node))
    {
        if (rw->filter_h() <= 16)
            return false;
        if (rw->padding_h() == padding::zero()
            && rw->padding_w() == padding::zero()
            && (size_t)rw->filter_h() == rw->input().shape()[2]
            && (size_t)rw->filter_w() == rw->input().shape()[3])
        {
            context.inputs.emplace_back(&rw->input());
            context.outputs.emplace_back(&rw->output());

            context.matched_nodes.emplace_back(rw);
            return true;
        }
    }

    return false;
}

void global_reduce_window_to_reduce_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_rw = static_cast<reduce_window2d &>(*context.matched_nodes[0]);

    // TODO: Handle activation
    assert(old_rw.fused_activation() == value_range<float>::full());
    auto rd = context.graph.emplace<reduce>(old_rw.reduce_op(), old_rw.input().shape(), axis_t { 2, 3 }, old_rw.init_value(), true);
    rd->name(old_rw.name());

    rd->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rd->output());
}

bool reduce_to_global_reduce_window_transform::on_try_match(node &node, transform_context &context)
{
    if (auto rd = node_cast<reduce>(node))
    {
        if (rd->input().shape().size() == 4
            && rd->axis().size() == 2 && rd->axis()[0] == 2 && rd->axis()[1] == 3
            && rd->input().shape()[2] <= 16)
        {
            context.inputs.emplace_back(&rd->input());
            context.outputs.emplace_back(&rd->output());

            context.matched_nodes.emplace_back(rd);
            return true;
        }
    }

    return false;
}

void reduce_to_global_reduce_window_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_rd = static_cast<reduce &>(*context.matched_nodes[0]);

    // TODO: Handle activation
    auto rw = context.graph.emplace<reduce_window2d>(old_rd.reduce_op(), old_rd.input().shape(), old_rd.init_value(), (int32_t)old_rd.input().shape()[2],
        (int32_t)old_rd.input().shape()[3], padding::zero(), padding::zero(), 1, 1, 1, 1, value_range<float>::full());
    rw->name(old_rd.name());
    auto rshape = context.graph.emplace<bitcast>(rw->output().type(), rw->output().shape(), old_rd.output().shape());
    rshape->name(old_rd.name() + "/rshape");

    rw->input().connect(output);
    rshape->input().connect(rw->output());
    for (auto &in : dup(inputs))
        in->connect(rshape->output());
}
