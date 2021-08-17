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
#pragma once
#include "buffer_allocator.h"
#include "liveness_analysis.h"
#include "schedule_types.h"
#include <filesystem>

namespace nncase
{
class target;
}

namespace nncase::schedule
{
class module_schedule_context;
class model_schedule_context;

struct caller_context
{
    lifetime_recorder &lifetime;
};

class function_schedule_context : public function_schedule_result
{
public:
    function_schedule_context(ir::graph &graph, module_schedule_context &mod_sched);
    function_schedule_context(const function_schedule_context &) = delete;
    function_schedule_context(function_schedule_context &&) = default;

    function_schedule_context &operator=(const function_schedule_context &) = delete;
    function_schedule_context &operator=(function_schedule_context &&) = default;

    const module_type_t &module_type() const noexcept { return graph->module_type(); }
    std::span<ir::output_node *> outputs() const noexcept { return outputs_; }
    std::unordered_map<const ir::output_connector *, logical_buffer *> &logical_buffer_map() noexcept { return logical_buffer_map_; }
    std::list<logical_buffer> &logical_buffers() noexcept { return logical_buffers_; }
    std::vector<physical_buffer> &physical_buffers() noexcept { return physical_buffers_; }

    void visit_function(caller_context &caller_ctx);
    void end_schedule();

private:
    void create_allocators();
    void generate_compute_sequence();
    void make_logical_buffers(caller_context &caller_ctx);
    void analyze_buffer_alias();
    void update_offset();
    void fix_lifetime();
    void make_physical_buffers();
    void allocate_physical_buffers();
    void assign_allocations();

    void dump(const std::filesystem::path &dump_dir);

private:
    module_schedule_context &mod_sched_;
    std::span<ir::output_node *> outputs_;
    allocator_map_t allocators_;
    std::vector<std::shared_ptr<buffer_allocator>> allocator_holder_;
    std::unordered_map<const ir::output_connector *, logical_buffer *> logical_buffer_map_;
    std::list<logical_buffer> logical_buffers_;
    std::vector<physical_buffer> physical_buffers_;
};

class module_schedule_context
{
public:
    module_schedule_context(module_schedule_result &result, model_schedule_context &model_sched, module_type_t type);
    module_schedule_context(const module_schedule_context &) = delete;
    module_schedule_context(module_schedule_context &&) = default;

    module_schedule_context &operator=(const module_schedule_context &) = delete;
    module_schedule_context &operator=(module_schedule_context &&) = default;

    module_schedule_result &module_result() const noexcept { return result_; }
    model_schedule_context &model_sched() const noexcept { return model_sched_; }
    allocator_map_t &allocators() noexcept { return allocators_; }
    buffer_allocator &shared_allocator(const module_type_t &type);

    void visit_function(ir::graph &graph, caller_context &caller_ctx);
    void end_schedule();

private:
    module_schedule_result &result_;
    model_schedule_context &model_sched_;
    module_type_t type_;
    allocator_map_t allocators_;
    std::vector<std::shared_ptr<buffer_allocator>> allocator_holder_;
    shared_allocator_map_t shared_allocators_;
    std::vector<function_schedule_context> functions_;
    std::filesystem::path dump_dir_;
};

class model_schedule_context
{
public:
    model_schedule_context(model_schedule_result &result, nncase::target &target, bool skip_buffer_alias);
    model_schedule_context(const model_schedule_context &) = delete;
    model_schedule_context(model_schedule_context &&) = default;

    model_schedule_context &operator=(const model_schedule_context &) = delete;
    model_schedule_context &operator=(model_schedule_context &&) = default;

    nncase::target &target() const noexcept { return target_; }
    bool skip_buffer_alias() const noexcept { return skip_buffer_alias_; }
    void config_dump(std::filesystem::path dump_dir);
    const std::filesystem::path &dump_dir() const noexcept { return dump_dir_; }
    model_schedule_result &model_result() const noexcept { return result_; }

    void schedule(ir::graph &entry_function);
    void visit_function(ir::graph &graph, caller_context &caller_ctx);

private:
    void end_schedule();

private:
    model_schedule_result &result_;
    nncase::target &target_;
    bool skip_buffer_alias_;
    std::filesystem::path dump_dir_;
    module_schedule_context *entry_module_;
    ir::graph *entry_function_;
    std::unordered_map<module_type_t, module_schedule_context> module_contexts_;
};
}
