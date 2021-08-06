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
#include <nncase/ir/graph.h>
#include <unordered_map>
#include <vector>

namespace nncase::schedule
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
struct module_schedule_result;

struct function_schedule_result
{
    ir::graph *graph;
    module_schedule_result *module;
    std::vector<ir::node *> compute_sequence;
};

struct module_schedule_result
{
    module_type_t type;
    std::vector<function_schedule_result> functions;
    std::unordered_map<ir::graph *, function_schedule_result *> functions_map;
    allocation_map_t allocations;
    std::unordered_map<memory_location_t, size_t> max_usages;
    std::unordered_map<module_type_t, size_t> shared_max_usages;
};

struct schedule_result
{
    std::vector<module_schedule_result> modules;
    function_schedule_result *entry_function;
};
}
