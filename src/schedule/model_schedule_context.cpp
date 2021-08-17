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
#include <nncase/schedule/schedule_context.h>
#include <nncase/targets/target.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::ir::transforms;

model_schedule_context::model_schedule_context(model_schedule_result &result, nncase::target &target, bool skip_buffer_alias)
    : result_(result), target_(target), skip_buffer_alias_(skip_buffer_alias), entry_module_(nullptr), entry_function_(nullptr)
{
}

void model_schedule_context::config_dump(std::filesystem::path dump_dir)
{
    dump_dir_ = std::move(dump_dir);
}

void model_schedule_context::schedule(ir::graph &entry_function)
{
    // 1. Calculate modules count
    auto reachable_graphs = entry_function.reachable_graphs();
    size_t modules_count;
    {
        std::unordered_set<module_type_t> modules;
        for (auto subgraph : reachable_graphs)
            modules.emplace(subgraph->module_type());
        modules_count = modules.size();
    }

    result_.modules.resize(modules_count);
    entry_function_ = &entry_function;

    // 2. Visit entry function
    std::list<logical_buffer> dummy_buffers;
    std::unordered_map<const ir::output_connector *, logical_buffer *> dummy_buffer_map;
    lifetime_recorder dummy_lifetime(dummy_buffers, dummy_buffer_map);
    caller_context dummy_ctx { dummy_lifetime };
    visit_function(entry_function, dummy_ctx);

    // 3. Collect schedule results
    end_schedule();
}

void model_schedule_context::visit_function(ir::graph &graph, caller_context &caller_ctx)
{
    auto it = module_contexts_.find(graph.module_type());
    if (it == module_contexts_.end())
    {
        size_t module_id = module_contexts_.size();
        it = module_contexts_.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(graph.module_type()),
                                 std::forward_as_tuple(result_.modules[module_id], *this, graph.module_type()))
                 .first;
    }

    it->second.visit_function(graph, caller_ctx);
    if (&graph == entry_function_)
        entry_module_ = &it->second;
}

void model_schedule_context::end_schedule()
{
    for (auto &module_p : module_contexts_)
        module_p.second.end_schedule();
    result_.entry_function = entry_module_->module_result().functions_map.at(entry_function_);
}
