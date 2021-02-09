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
#pragma once
#include "buffer_allocator.h"
#include "buffers.h"
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
        ir::shape_t parent_shape;

        size_t linear_end() const noexcept { return start + size; }

        bool overlap(const buffer_allocation &rhs) const noexcept
        {
            return size != 0 && rhs.size != 0 && memory_location == rhs.memory_location && (start < rhs.linear_end() && linear_end() > rhs.start);
        }

        memory_range runtime_type() const
        {
            return { memory_location, type, (uint32_t)start, (uint32_t)size };
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
        std::unordered_map<ir::graph *, module_schedule_result> modules;
        ir::graph *main_module;
    };

    struct schedule_context;

    class scheduler
    {
    public:
        scheduler(target &target, ir::graph &main_graph, std::span<ir::output_node *> outputs)
            : target_(target), main_graph_(main_graph), outputs_(outputs) { }

        schedule_result schedule();

    private:
    private:
        target &target_;
        ir::graph &main_graph_;
        std::span<ir::output_node *> outputs_;
        schedule_context *cnt_context_;
    };
}
}
