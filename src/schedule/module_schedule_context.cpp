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

module_schedule_context::module_schedule_context(module_schedule_result &result, model_schedule_context &model_sched, module_type_t type)
    : result_(result), model_sched_(model_sched), type_(type)
{
    result_.type = type;
    model_sched.target().register_allocators(type_, allocators_, allocator_holder_);
}

buffer_allocator &module_schedule_context::shared_allocator(const module_type_t &type)
{
    return *shared_allocators_.at(type);
}

void module_schedule_context::visit_function(ir::graph &graph, caller_context &caller_ctx)
{
    auto &fctx = functions_.emplace_back(graph, *this);
    fctx.visit_function(caller_ctx);
}

void module_schedule_context::end_schedule()
{
    for (auto &allocator : allocators_)
    {
        allocator.second->finish();

        if (allocator.first != mem_input
            && allocator.first != mem_output)
            module_result().max_usages.emplace(allocator.first, allocator.second->max_usage());
    }

    result_.functions.resize(functions_.size());
    for (size_t i = 0; i < functions_.size(); i++)
    {
        functions_[i].end_schedule();
        auto &func = result_.functions[i] = function_schedule_result { functions_[i] };
        result_.functions_map.emplace(func.graph, &func);
    }
}
