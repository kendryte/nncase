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
#include "liveness_analysis.h"
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/schedule_context.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/optimize_allocation.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::ir::transforms;

schedule_context::schedule_context(nncase::target &target, const allocator_map_t &allocators, bool skip_buffer_alias)
    : target_(target), allocators_(allocators), skip_buffer_alias_(skip_buffer_alias)
{
}

void schedule_context::generate_compute_sequence()
{
    std::unordered_set<node *> used_inputs;
    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
        if (node.runtime_opcode() == op_input_node)
            used_inputs.emplace(&node);
        else if (node.attributes() & node_attr_action)
            compute_sequence.emplace_back(&node);
    });

    alloc_visitor.visit(outputs_);

    size_t i = 0;
    for (auto in : graph->inputs())
    {
        if (used_inputs.contains(in))
        {
            compute_sequence.insert(compute_sequence.begin() + i, in);
            i++;
        }
    }
}

void schedule_context::make_logical_buffers()
{
    lifetime_recorder lr(logical_buffers_, logical_buffer_map_, skip_buffer_alias_);
    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
        for (auto out : node.outputs())
            lr.allocate(*out);

        lr.grow_age();

        for (auto in : node.inputs())
        {
            auto out = in->connection();
            assert(out);
            lr.release(*out);
        }
    });
    alloc_visitor.visit(outputs_);
}

void schedule_context::analyze_buffer_alias()
{
    pass_manager pmgr(*graph, target_);
    pmgr.add_pass<alias_bitcast_buffer_pass>();
    pmgr.add_pass<alias_concat_buffer_pass>();
    pmgr.run();
}

void schedule_context::update_offset()
{
    auto visitor = make_relay_ir_visitor<bfs_ir_pre_order_visitor>([&](node &node) {
        if (auto b = node_cast<bitcast>(node))
        {
            if (b->attributes() & node_attr_action)
                return;

            auto &in_buf = logical_buffer_map_.at(b->input().connection());
            auto &out_buf = logical_buffer_map_.at(&b->output());

            if (in_buf->memory_location() == mem_data && in_buf->parent() && out_buf->parent())
            {
                in_buf->parent()->offset += out_buf->parent()->offset;
            }
        }
        else if (auto c = node_cast<concat>(node))
        {
            if (c->attributes() & node_attr_action)
                return;

            auto &out_buf = logical_buffer_map_.at(&c->output());

            for (auto &in : c->inputs())
            {
                auto in_buf = logical_buffer_map_.at(in->connection());
                if (in_buf->parent() && out_buf->parent())
                {
                    in_buf->parent()->offset += out_buf->parent()->offset;
                }
            }
        }
        // TODO: slice
        // else if (auto s = node_cast<slice>(node))
        // {
        // }
    });
    visitor.visit(outputs_);
}

void schedule_context::fix_lifetime()
{
    // Assign parent
    for (auto &bp : logical_buffers_)
    {
        auto &p = bp.parent();
        if (p)
        {
            auto parent = p->parent;
            while (parent->parent())
                parent = parent->parent()->parent;
            p->parent = parent;
        }
    }

    // Extend lifetime
    for (auto &bp : logical_buffers_)
    {
        auto &lifetime = bp.lifetime();
        if (bp.parent())
        {
            auto &p_liftime = bp.parent()->parent->lifetime();
            auto birth = std::min(lifetime.birth, p_liftime.birth);
            auto end = std::max(lifetime.end(), p_liftime.end());
            p_liftime.birth = birth;
            p_liftime.age = end - birth;
        }
    }
}

void schedule_context::make_physical_buffers()
{
    std::unordered_map<logical_buffer *, size_t> physical_ids;
    for (auto &bp : logical_buffers_)
    {
        if (!bp.parent())
        {
            auto id = physical_ids.size();
            physical_ids.emplace(&bp, id);
            auto &pb = physical_buffers_.emplace_back(id, bp);
            if (auto c = node_cast<constant>(bp.owner().owner()))
                pb.alignment(c->alignment());
        }
    }

    // Assign parents
    for (auto &bp : logical_buffers_)
    {
        auto parent = bp.parent() ? bp.parent()->parent : &bp;
        bp.physical() = &physical_buffers_.at(physical_ids.at(parent));
    }
}

void schedule_context::allocate_physical_buffers()
{
    allocator_map_t new_allocators;
    std::vector<std::shared_ptr<buffer_allocator>> new_allocator_holder;
    target_.register_allocators(module_type(), new_allocators, new_allocator_holder);

    // Input & output don't actually allocate inside the module, they are passed from the caller
    // They just need relative offset, so don't inherit previous allocators
    for (auto allocator : allocators_)
    {
        if (allocator.first != mem_input
            && allocator.first != mem_output)
            new_allocators[allocator.first] = allocator.second;
    }

    std::vector<physical_buffer *> orders;
    orders.reserve(physical_buffers_.size());
    for (auto &b : physical_buffers_)
        orders.emplace_back(&b);
    std::sort(orders.begin(), orders.end(), [](const physical_buffer *lhs, const physical_buffer *rhs) { return lhs->lifetime().birth < rhs->lifetime().birth; });

    for (auto &b : orders)
        new_allocators.at(b->owner().memory_location())->mark(*b);

    for (auto &alloc : new_allocators)
    {
        alloc.second->finish();
        max_usages.emplace(alloc.first, alloc.second->max_usage());
    }

    for (auto &b : physical_buffers)
        b.allocation() = memory_span { allocators.at(b.owner().memory_location())->allocations().at(&b) };
}
