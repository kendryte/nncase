/* Copyright 2019-2020 Canaan Inc.
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
#include <algorithm>
#include <llir/transforms/pass.h>
#include <llir/visitor.h>

using namespace nncase;
using namespace nncase::llir;
using namespace nncase::llir::transforms;

namespace
{
class transform_apply_visitor : public dfs_ir_visitor
{
public:
    using dfs_ir_visitor::visit;
    llir::graph *graph;
    nncase::target *target;
    bool need_retry = false;
    transforms::transform *transform;

protected:
    bool visit(node &node) override
    {
        transform_context context { *graph, *target };
        if (transform->try_match(node, context))
        {
            transform->process(context);
            need_retry = true;
            return true;
        }

        return false;
    }
};
}

void pass::run(graph &graph, target &target)
{
    transform_apply_visitor visitor;
    visitor.graph = &graph;
    visitor.target = &target;
    bool next_pass = false;

    do
    {
        next_pass = false;

        for (auto &&transform : transforms_)
        {
            visitor.transform = transform.get();
            visitor.need_retry = false;
            visitor.visit(graph);

            if (visitor.need_retry)
            {
                next_pass = true;
                graph.collect();
                break;
            }
        }
    } while (next_pass);
}

void pass_manager::run()
{
    for (auto &&pass : passes_)
        pass.run(graph_, target_);
}
