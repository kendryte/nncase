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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/optimize_benchmark.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

void optimize_benchmark_pass::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    auto alias_visitor = make_relay_ir_visitor([&](node &node)
        {
            if (auto c = node_cast<constant>(node))
            {
                auto inputs = c->output().connections();
                if (std::all_of(inputs.begin(), inputs.end(), [](input_connector *conn)
                        { return !(conn->attributes() & cnctr_attr_no_dummy_for_benchmark); }))
                    c->output().memory_location(mem_data);
            }
        });
    alias_visitor.visit(graph);
}
