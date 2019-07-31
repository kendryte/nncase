#pragma once
#include <datatypes.h>
#include <iostream>
#include <ir/graph.h>
#include <runtime/binary_writer.h>
#include <runtime/runtime_op.h>
#include <scheduler/scheduler.h>

namespace nncase
{
namespace codegen
{
    class codegen_context
    {
    public:
        codegen_context(std::ostream &output, const std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, const std::unordered_map<ir::output_connector *, scheduler::memory_allocation> &allocations);

        memory_range get_allocation(ir::input_connector &conn) const
        {
            return get_allocation(*conn.connection());
        }

        memory_range get_allocation(ir::output_connector &conn) const;
        uint32_t memory_usage() const noexcept { return (uint32_t)allocators_.at(mem_main)->max_usage(); }
        uint32_t constant_usage() const noexcept { return (uint32_t)allocators_.at(mem_const)->max_usage(); }

        runtime::binary_writer &writer() noexcept { return writer_; }

    private:
        runtime::binary_writer writer_;
        const std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators_;
        const std::unordered_map<ir::output_connector *, scheduler::memory_allocation> &allocations_;
    };

    struct node_body
    {
        virtual runtime::runtime_opcode opcode() const noexcept = 0;
        virtual void serialize(runtime::binary_writer &writer) = 0;
    };

    template <runtime::runtime_opcode Op, class T>
    struct node_body_impl : public T, public node_body
    {
        runtime::runtime_opcode opcode() const noexcept override
        {
            return Op;
        }

        void serialize(runtime::binary_writer &writer) override
        {
            T::serialize(writer);
        }
    };

    using emitter_t = std::function<std::unique_ptr<node_body>(ir::node &node, codegen_context &context)>;

    void register_emitter(ir::node_opcode opcode, emitter_t emitter);
    void disable_emitter(ir::node_opcode opcode);
    void gencode(codegen_context &context, xtl::span<ir::node *> compute_sequence);
}
}
