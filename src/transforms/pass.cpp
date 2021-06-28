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
#include <algorithm>
#include <filesystem>
#include <nncase/ir/debug.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/pass.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

namespace
{
class transform_apply_visitor : public dfs_ir_post_order_visitor
{
public:
    using dfs_ir_post_order_visitor::visit;
    ir::graph *graph;
    ir::quantizer *quantizer;
    nncase::target *target;
    std::optional<std::filesystem::path> dump_dir;
    bool need_retry = false;
    transforms::transform *transform;

protected:
    bool visit(node &node) override
    {
        auto context = transform->create_context(*graph, *target);
        context->quantizer = quantizer;
        context->dump_dir = dump_dir;

        if (transform->try_match(node, *context))
        {
            transform->process(*context);
            need_retry = true;
            return true;
        }

        return false;
    }
};
}

void pass::run(graph &graph, target &target, const run_pass_options &options)
{
    run_core(graph, target, options);
    graph.cse();
    if (options.dump_dir)
    {
        auto dump_path = *options.dump_dir / "passes" / dump_name_;
        std::filesystem::create_directories(dump_path);
        ir::dump_graph(graph, dump_path);
    }
}

void transform_pass::run_core(graph &graph, target &target, const run_pass_options &options)
{
    transform_apply_visitor visitor;
    visitor.graph = &graph;
    visitor.target = &target;
    visitor.dump_dir = options.dump_dir;
    visitor.quantizer = options.quantizer;
    bool next_pass = false;

    do
    {
        next_pass = false;

        for (size_t idx = 0; idx < transforms_.size(); idx++)
        {
            auto &&transform = transforms_[idx];
            visitor.transform = transform.get();
            visitor.need_retry = false;
            visitor.visit(graph);

            if (visitor.need_retry)
            {
                next_pass = true;
                graph.dce();
                break;
            }
        }
    } while (next_pass);
}

void pass_manager::dump_dir(const std::filesystem::path &dir)
{
    dump_dir_ = dir;
}

void pass_manager::quantizer(ir::quantizer *q)
{
    quantizer_ = q;
}

void pass_manager::schedule_context(schedule::schedule_context *c)
{
    schedule_context_ = c;
}

void pass_manager::run()
{
    run_pass_options options;
    options.dump_dir = dump_dir_;
    options.quantizer = quantizer_;
    options.schedule_context = schedule_context_;

    for (auto &pass : passes_)
        pass->run(graph_, target_, options);
}
