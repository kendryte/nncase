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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/scheduler.h>
#include <nncase/transforms/neutral/optimize_allocation.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;
using namespace nncase::schedule;

void mark_no_action_concat_pass::run_core(graph &graph, nncase::target &target, const run_pass_options &options)
{
    auto &context = *options.schedule_context;
    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
        if (auto c = node_cast<concat>(node))
        {
            auto inputs = c->inputs();
            auto outputs = c->output().connections();

            // 1. concat by outer-most axis
            auto is_simple_concat = (c->axis() == 0 || std::all_of(inputs[0]->shape().begin(), inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
            auto &out_buf = context.logical_buffers.at(&c->output());
        }
    });
    alias_visitor.visit(context.outputs);
}
