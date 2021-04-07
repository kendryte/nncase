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
class transform_apply_visitor : public dfs_ir_visitor
{
public:
    using dfs_ir_visitor::visit;
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

void pass::run(graph &graph, target &target, ir::quantizer *quantizer, std::optional<std::filesystem::path> dump_dir)
{
    transform_apply_visitor visitor;
    visitor.graph = &graph;
    visitor.target = &target;
    visitor.dump_dir = dump_dir;
    visitor.quantizer = quantizer;
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

void pass_manager::add_pass(pass &&pass)
{
    passes_.emplace_back(std::move(pass));
}

void pass_manager::dump_dir(const std::filesystem::path &dir)
{
    dump_dir_ = dir;
}

void pass_manager::quantizer(ir::quantizer *q)
{
    quantizer_ = q;
}

void pass_manager::run()
{
    for (auto &&pass : passes_)
    {
        pass.run(graph_, target_, quantizer_, dump_dir_);
        if (dump_dir_)
        {
            auto dump_path = *dump_dir_ / "passes" / pass.name();
            std::filesystem::create_directories(dump_path);
            ir::dump_graph(graph_, dump_path);
        }
        graph_.cse();
    }
}
