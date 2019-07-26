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

    private:
        const std::unordered_map<memory_type_t, memory_allocator *> &allocators_;
        std::unordered_map<ir::output_connector *, memory_node *> memory_map_;
        std::unordered_map<ir::output_connector *, memory_allocation> allocations_;
    };

    void register_input_allocator(ir::node_opcode opcode, std::function<void(ir::node &, ir::output_connector &, allocation_context)> allocator);
    void schedule(xtl::span<ir::output_node *> outputs, allocation_context &context, std::vector<ir::node *> &compute_sequence);
}
}
