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
#include <fmt/format.h>
#include <fstream>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/call.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/schedule_context.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/optimize_allocation.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::ir::transforms;

namespace
{
memory_location_t decide_memory_location(ir::output_connector &conn, bool skip_buffer_alias) noexcept
{
    auto &opcode = conn.owner().runtime_opcode();
    if (opcode == op_input_node)
        return mem_input;
    else if (opcode == op_constant)
        return conn.memory_location();

    auto inputs = conn.connections();
    if (skip_buffer_alias)
    {
        if (std::any_of(inputs.begin(), inputs.end(), [](input_connector *conn) { return conn->owner().runtime_opcode() == op_output_node; }))
            return mem_output;
    }

    //if (opcode == op_call && conn.memory_location() == mem_data)
    //    return mem_shared_data;
    //if (conn.memory_location() == mem_data
    //    && std::any_of(inputs.begin(), inputs.end(), [](input_connector *conn) { return conn->owner().runtime_opcode() == op_call; }))
    //    return mem_shared_data;

    return conn.memory_location();
}
}

function_schedule_context::function_schedule_context(ir::graph &graph, module_schedule_context &mod_sched)
    : mod_sched_(mod_sched)
{
    this->graph = &graph;
    this->outputs_ = graph.outputs();
    this->module = &mod_sched.module_result();

    create_allocators();
}

void function_schedule_context::create_allocators()
{
    mod_sched_.model_sched().target().register_allocators(module_type(), allocators_, allocator_holder_);

    // Input & output don't actually allocate inside the module, they are passed from the caller
    // They just need relative offset, so don't inherit previous allocators
    for (auto allocator : mod_sched_.allocators())
    {
        if (allocator.first != mem_input
            && allocator.first != mem_output)
            allocators_[allocator.first] = allocator.second;
    }
}

void function_schedule_context::visit_function(caller_context &caller_ctx)
{
    make_logical_buffers(caller_ctx);
    if (!mod_sched_.model_sched().skip_buffer_alias())
        analyze_buffer_alias();
    update_offset();
    fix_lifetime();
    generate_compute_sequence();
    make_physical_buffers();

    allocate_physical_buffers();
}

void function_schedule_context::end_schedule()
{
    for (auto &allocator : allocator_holder_)
    {
        allocator->finish();

        if (allocator.get() == allocators_.at(mem_input))
            input_pool_size = allocator->max_usage();
        else if (allocator.get() == allocators_.at(mem_output))
            output_pool_size = allocator->max_usage();
    }

    assign_allocations();

    auto dump_dir = mod_sched_.model_sched().dump_dir();
    if (!dump_dir.empty())
        dump(dump_dir);
}

void function_schedule_context::generate_compute_sequence()
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

void function_schedule_context::make_logical_buffers(caller_context &caller_ctx)
{
    auto skip_buffer_alias = mod_sched_.model_sched().skip_buffer_alias();
    lifetime_recorder lr(logical_buffers_, logical_buffer_map_);

    // 1. Adjust base age to caller's age
    lr.current_age(caller_ctx.lifetime.current_age());

    // 2. Estimate buffer lifetime
    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
        for (auto out : node.outputs())
            lr.allocate(*out, decide_memory_location(*out, skip_buffer_alias));

        lr.grow_age();

        if (auto c = node_cast<call>(node))
        {
            caller_context new_caller_ctx { lr };
            mod_sched_.model_sched().visit_function(c->target(), new_caller_ctx);
        }

        for (auto in : node.inputs())
        {
            auto out = in->connection();
            assert(out);
            lr.release(*out);
        }
    });
    alloc_visitor.visit(outputs_);

    // 3. Adjust caller's age to now
    caller_ctx.lifetime.current_age(lr.current_age());
}

void function_schedule_context::analyze_buffer_alias()
{
    pass_manager pmgr(*graph, mod_sched_.model_sched().target());
    pmgr.schedule_context(this);
    pmgr.add_pass<alias_bitcast_buffer_pass>();
    pmgr.add_pass<alias_concat_buffer_pass>();
    pmgr.add_pass<alias_slice_buffer_pass>();
    pmgr.run();
}

void function_schedule_context::update_offset()
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
        else if (auto s = node_cast<slice>(node))
        {
            if (!(s->attributes() & node_attr_action))
            {
                auto &in_buf = logical_buffer_map.at(s->input().connection());
                auto &out_buf = logical_buffer_map.at(&s->output());

                in_buf->parent()->offset += out_buf->parent()->offset;
            }
        }
    });
    visitor.visit(outputs_);
}

void function_schedule_context::fix_lifetime()
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

void function_schedule_context::make_physical_buffers()
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

void function_schedule_context::allocate_physical_buffers()
{
    std::vector<physical_buffer *> orders;
    orders.reserve(physical_buffers_.size());
    for (auto &b : physical_buffers_)
        orders.emplace_back(&b);
    std::sort(orders.begin(), orders.end(), [](const physical_buffer *lhs, const physical_buffer *rhs) { return lhs->lifetime().birth < rhs->lifetime().birth; });

    for (auto &b : orders)
    {
        auto location = b->owner().memory_location();
        if (location != mem_shared_data)
            allocators_.at(location)->mark(*b);
        else
            mod_sched_.shared_allocator(b->owner().shared_module()).mark(*b);
    }
}

void function_schedule_context::assign_allocations()
{
    for (auto &b : physical_buffers_)
        b.allocation() = memory_span { allocators_.at(b.owner().memory_location())->allocations().at(&b) };

    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
        for (auto out : node.outputs())
        {
            auto &lbuf = *logical_buffer_map_.at(out);
            auto &owner = lbuf.physical()->owner();
            auto &memory = lbuf.physical()->allocation();

            // TODO: take account of subbuffer
            buffer_allocation alloc {};
            alloc.memory_location = owner.memory_location();
            alloc.type = lbuf.type();
            alloc.size = allocators_.at(alloc.memory_location)->get_size_in_bytes(lbuf);
            alloc.shape = lbuf.shape();
            assert(lbuf.strides_shape().size());
            alloc.strides_shape = lbuf.strides_shape();
            alloc.strides = to_strides(alloc.strides_shape);
            alloc.start = memory.start;
            if (lbuf.parent())
            {
                alloc.start += lbuf.parent()->offset;
            }

            module->allocations.emplace(out, alloc);
        }
    });
    alloc_visitor.visit(outputs_);
}

void function_schedule_context::dump(const std::filesystem::path &dump_dir)
{
    std::ofstream writer(dump_dir / (graph->escaped_name() + ".sched"));

    auto fmt_shape = [&](const logical_buffer &buf) {
        auto alloc = module->allocations.at(&buf.owner());
        return fmt::format("<{} {} {} bytes of {}>",
            datatype_names(buf.type()),
            ir::to_string(buf.shape()),
            alloc.size,
            ir::to_string(buf.strides_shape()));
    };

    // 1. allocation
    writer << ".physical_buffer" << std::endl;
    for (auto &buf : physical_buffers_)
    {
        auto alloc = buf.allocation();

        writer << fmt::format("%{} : {} @{}[{}, {}]",
            buf.id(),
            fmt_shape(buf.owner()),
            to_string(buf.owner().memory_location()),
            alloc.start,
            alloc.end())
               << std::endl;
    }

    // 2. compute sequence
    writer << std::endl
           << ".compute_sequence" << std::endl;

    //auto print_inputs = [&]()

    // 2.1 function name
    writer << "fn " << graph->escaped_name() << "(";

    // 2.2 inputs
    {
        bool comma = false;
        for (auto in : graph->inputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *logical_buffer_map_.at(&in->output());
            auto &pbuf = *lbuf.physical();
            writer << '%' << pbuf.id();
        }
    }

    writer << ") : (";

    // 2.2 input shapes
    {
        bool comma = false;
        for (auto in : graph->inputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *logical_buffer_map_.at(&in->output());
            writer << fmt_shape(lbuf);
        }
    }

    writer << ") -> (";
    // 2.3 output shapes
    {
        bool comma = false;
        for (auto in : graph->outputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *logical_buffer_map_.at(in->input().connection());
            writer << fmt_shape(lbuf);
        }
    }

    writer << ")" << std::endl;

#define IDENT "    "

    // 2.4 body
    writer << '{' << std::endl;

    for (auto node : compute_sequence)
    {
        // 2.4.1 outputs
        if (node->runtime_opcode() != op_output_node)
        {
            bool comma = false;
            for (auto out : node->outputs())
            {
                if (!comma)
                    comma = true;
                else
                    writer << ", ";

                auto &lbuf = *logical_buffer_map_.at(out);
                auto &pbuf = *lbuf.physical();
                auto &alloc = module->allocations.at(out);
                auto pbuf_alloc = pbuf.allocation();
                if (alloc.size == pbuf_alloc.size)
                    writer << IDENT << fmt::format("%{}", pbuf.id());
                else
                    writer << IDENT << fmt::format("%{}[{}, {}]", pbuf.id(), alloc.start - pbuf_alloc.start, alloc.linear_end() - pbuf_alloc.start);
            }

            // 2.4.2 inst name
            writer << " = " << node->runtime_opcode().name << "(";
        }
        else
        {
            writer << IDENT << "return ";
        }

        // 2.4.3 inputs
        {
            bool comma = false;
            for (auto in : node->inputs())
            {
                if (!comma)
                    comma = true;
                else
                    writer << ", ";

                auto &lbuf = *logical_buffer_map_.at(in->connection());
                auto &pbuf = *lbuf.physical();
                auto &alloc = module->allocations.at(in->connection());
                auto pbuf_alloc = pbuf.allocation();
                if (alloc.size == pbuf.allocation().size)
                    writer << fmt::format("%{}", pbuf.id());
                else
                    writer << fmt::format("%{}[{}, {}]",
                        pbuf.id(),
                        alloc.start - pbuf_alloc.start,
                        alloc.linear_end() - pbuf_alloc.start);
            }
        }

        if (node->runtime_opcode() == op_output_node)
        {
            writer << " : (";
        }
        else
        {
            writer << ") : (";
        }

        // 2.4.4 input shapes
        {
            bool comma = false;
            for (auto in : node->inputs())
            {
                if (!comma)
                    comma = true;
                else
                    writer << ", ";

                auto &lbuf = *logical_buffer_map_.at(in->connection());
                writer << fmt_shape(lbuf);
            }
        }

        if (node->runtime_opcode() == op_output_node)
        {
            writer << ")" << std::endl;
            continue;
        }
        else
        {
            writer << ") -> (";
        }

        // 2.4.5 output shapes
        {
            bool comma = false;
            for (auto out : node->outputs())
            {
                if (!comma)
                    comma = true;
                else
                    writer << ", ";

                auto &lbuf = *logical_buffer_map_.at(out);
                writer << fmt_shape(lbuf);
            }
        }

        writer << ")" << std::endl;
    }

    writer << '}' << std::endl;
}
