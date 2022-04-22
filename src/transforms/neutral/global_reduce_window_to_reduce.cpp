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
    auto rd = context.graph.emplace<reduce>(old_rw.reduce_op(), output.type(), old_rw.input().shape(), axis_t { 2, 3 }, old_rw.init_value(), true);
    rd->name(old_rw.name());

    rd->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rd->output());
}
