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
#include <ir/op_utils.h>
#include <ir/ops/constant.h>
#include <ir/visitor.h>
#include <scheduler/scheduler.h>
#include <unordered_map>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::scheduler;

namespace
{
std::unordered_map<node_opcode, std::function<void(ir::node &, ir::output_connector &, allocation_context)>> g_allocators;
}

void nncase::scheduler::register_input_allocator(node_opcode opcode, std::function<void(ir::node &, ir::output_connector &, allocation_context)> allocator)
{
    g_allocators.emplace(opcode, std::move(allocator));
}

allocation_context::allocation_context(const std::unordered_map<memory_type_t, memory_allocator *> &allocators)
    : allocators_(allocators)
{
}

void allocation_context::allocate_default(ir::output_connector &conn)
{
    auto allocator = allocators_.find(conn.memory_type());
    if (allocator == allocators_.end())
        throw std::runtime_error("Allocator is not found");

    auto it = memory_map_.find(&conn);
    if (it == memory_map_.end())
    {
        auto size = allocator->second->get_bytes(conn.type(), conn.shape());
        auto &node = allocator->second->allocate(size);
        memory_map_.emplace(&conn, &node);
    }
    else
    {
        it->second->add_ref();
    }
}

void allocation_context::release(ir::output_connector &conn)
{
    auto node = memory_map_.find(&conn);
    if (node != memory_map_.end())
        node->second->release();
}

void allocation_context::grow_age()
{
    for (auto &a : allocators_)
        a.second->grow_age();
}

void allocation_context::finish(uint32_t max_solve_secs)
{
    for (auto &a : allocators_)
        a.second->finish(max_solve_secs);

    for (auto &map : memory_map_)
    {
        auto conn = map.first;
        auto node = map.second;
        allocations_.emplace(conn, memory_allocation { conn->memory_type(), node->start(), node->valid_size() });
    }
}

void nncase::scheduler::schedule(xtl::span<output_node *> outputs, allocation_context &context, std::vector<ir::node *> &compute_sequence, uint32_t max_solve_secs)
{
    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
        for (auto &&out : node.outputs())
        {
            for (auto &&in : out.connections())
            {
                auto &in_node = in->owner();
                auto it = g_allocators.find(in_node.runtime_opcode());
                if (it != std::end(g_allocators))
                {
                    it->second(in_node, out, context);
                }
                else
                {
                    context.allocate_default(out);
                }
            }
        }

        compute_sequence.emplace_back(&node);
        context.grow_age();

        // Pin output
        if (node.runtime_opcode() != op_output_node)
        {
            for (auto &&in : node.inputs())
            {
                auto out = in.connection();
                assert(out);

                // Pin constant and input
				// TODO: Fix input pin
                if (out->memory_type() != mem_const/* && out->owner().runtime_opcode() != op_input_node*/)
                {
                    context.release(*out);
                }
            }
        }
    });

    alloc_visitor.visit(outputs);
    context.finish(max_solve_secs);

    auto check_visitor = make_relay_ir_visitor([&](node &node) {
        // check overlap
        {
            std::vector<memory_allocation> inputs, outputs;
            for (auto &&out : node.outputs())
                outputs.emplace_back(context.allocations().at(&out));

            for (auto &&in : node.inputs())
                inputs.emplace_back(context.allocations().at(in.connection()));

            //for (auto &&m : inputs)
            //{
            //    assert(std::none_of(outputs.begin(), outputs.end(), [&](const memory_allocation rhs) {
            //        return rhs.overlap(m);
            //    }));
            //}
        }
    });

    check_visitor.visit(outputs);
}
