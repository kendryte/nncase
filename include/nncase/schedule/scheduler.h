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
#include "buffers.h"
#include <filesystem>
#include <functional>
#include <nncase/ir/graph.h>
#include <span>
#include <unordered_map>
#include <vector>

namespace nncase
{
class target;

namespace schedule
{
    struct buffer_allocation
    {
        memory_location_t memory_location;
        datatype_t type;
        size_t start;
        size_t size;
        ir::shape_t shape;
        ir::shape_t strides;
        ir::shape_t strides_shape;

        size_t linear_end() const noexcept { return start + size; }

        bool overlap(const buffer_allocation &rhs) const noexcept
        {
            return size != 0 && rhs.size != 0 && memory_location == rhs.memory_location && (start < rhs.linear_end() && linear_end() > rhs.start);
        }

        memory_range runtime_type() const
        {
            return { .memory_location = memory_location, .datatype = type, .start = (uint32_t)start, .size = (uint32_t)size };
        }
    };

    using allocation_map_t = std::unordered_map<const ir::output_connector *, buffer_allocation>;

    struct module_schedule_result
    {
        ir::graph *graph;
        std::vector<ir::node *> compute_sequence;
        std::unordered_map<memory_location_t, size_t> max_usages;
        allocation_map_t allocations;
    };

    struct schedule_result
    {
        std::vector<ir::graph *> graph_orders;
        std::unordered_map<ir::graph *, module_schedule_result> modules;
        ir::graph *main_module;
    };

    struct schedule_context : module_schedule_result
    {
        bool skip_buffer_alias = false;
        nncase::target *target;
        module_type_t module_type;
        std::span<ir::output_node *> outputs;
        std::unordered_map<const ir::output_connector *, logical_buffer *> logical_buffer_map;
        std::list<logical_buffer> logical_buffers;
        std::vector<physical_buffer> physical_buffers;

        void generate_compute_sequence();
        void make_logical_buffers();
        void analyze_buffer_alias();
        void fix_concat_indices();
        void update_begin();
        void fix_lifetime();
        void make_physical_buffers();
        void allocate_physical_buffers();
        void assign_allocations();
    };

    class NNCASE_API scheduler
    {
    public:
        scheduler(target &target, ir::graph &main_graph, std::span<ir::output_node *> outputs)
            : target_(target), main_graph_(main_graph), outputs_(outputs) { }

        schedule_result schedule(bool skip_buffer_alias = false);
        void config_dump(std::filesystem::path dump_dir);

    private:
        void dump_schedule(const schedule_context &context);

    private:
        target &target_;
        ir::graph &main_graph_;
        std::span<ir::output_node *> outputs_;
        std::filesystem::path dump_dir_;
    };
}
}
