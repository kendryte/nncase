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
#include "schedule_types.h"

namespace nncase
{
class target;
}

namespace nncase::schedule
{
class schedule_context : public function_schedule_result
{
public:
    schedule_context(nncase::target &target, const allocator_map_t &allocators, bool skip_buffer_alias = false);

    bool skip_buffer_alias() const noexcept { return skip_buffer_alias_; }
    nncase::target &target() const noexcept { return target_; }
    const module_type_t &module_type() const noexcept { return graph->module_type(); }
    std::span<ir::output_node *> outputs() const noexcept { return outputs_; }
    std::unordered_map<const ir::output_connector *, logical_buffer *> &logical_buffer_map() noexcept { return logical_buffer_map_; }
    std::list<logical_buffer> &logical_buffers() noexcept { return logical_buffers_; }
    std::vector<physical_buffer> &physical_buffers() noexcept { return physical_buffers_; }

    void generate_compute_sequence();
    void make_logical_buffers();
    void analyze_buffer_alias();
    void update_offset();
    void fix_lifetime();
    void make_physical_buffers();
    void allocate_physical_buffers();
    void assign_allocations();

private:
    nncase::target &target_;
    bool skip_buffer_alias_;
    std::span<ir::output_node *> outputs_;
    const allocator_map_t &allocators_;
    std::unordered_map<const ir::output_connector *, logical_buffer *> logical_buffer_map_;
    std::list<logical_buffer> logical_buffers_;
    std::vector<physical_buffer> physical_buffers_;
};
}
