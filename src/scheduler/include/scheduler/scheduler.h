/* Copyright 2019 Canaan Inc.
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
#include "memory_allocator.h"
#include <functional>
#include <ir/graph.h>
#include <unordered_map>
#include <vector>

namespace nncase
{
namespace scheduler
{
    class allocation_context
    {
    public:
        allocation_context(const std::unordered_map<memory_type_t, memory_allocator *> &allocators);

        const std::unordered_map<ir::output_connector *, memory_allocation> &allocations() const noexcept { return allocations_; }

        void allocate_default(ir::output_connector &conn);
        void release(ir::output_connector &conn);
        void grow_age();
        void finish(uint32_t max_solve_secs);

    private:
        const std::unordered_map<memory_type_t, memory_allocator *> &allocators_;
        std::unordered_map<ir::output_connector *, memory_node *> memory_map_;
        std::unordered_map<ir::output_connector *, memory_allocation> allocations_;
    };

    void register_input_allocator(ir::node_opcode opcode, std::function<void(ir::node &, ir::output_connector &, allocation_context)> allocator);
    void schedule(xtl::span<ir::output_node *> outputs, allocation_context &context, std::vector<ir::node *> &compute_sequence, uint32_t max_solve_secs);
}
}
