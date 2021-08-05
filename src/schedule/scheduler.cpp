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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/schedule/scheduler.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/neutral/optimize_allocation.h>
#include <nncase/transforms/pass.h>
#include <unordered_map>
#include <xtensor/xarray.hpp>

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

    if (skip_buffer_alias)
    {
        auto connections = conn.connections();
        if (std::any_of(connections.begin(), connections.end(), [](input_connector *conn)
                { return conn->owner().runtime_opcode() == op_output_node; }))
            return mem_output;
    }

    return conn.memory_location();
}

class lifetime_recorder
{
public:
    lifetime_recorder(std::list<logical_buffer> &buffers, std::unordered_map<const ir::output_connector *, logical_buffer *> &buffer_map, bool skip_buffer_alias)
        : buffers_(buffers), buffer_map_(buffer_map), skip_buffer_alias_(skip_buffer_alias)
    {
    }

    void allocate(ir::output_connector &conn)
    {
        auto it = buffer_map_.find(&conn);
        if (it == buffer_map_.end())
        {
            logical_buffer buffer(next_buffer_id_++, conn, decide_memory_location(conn, skip_buffer_alias_));
            buffer.lifetime().birth = cnt_age_;
            buffer.lifetime().used_count = conn.connections().size();
            buffer.strides_shape() = buffer.shape();
            buffer_map_.emplace(&conn, &buffers_.emplace_back(buffer));
        }
    }

    void release(ir::output_connector &conn)
    {
        auto node = buffer_map_.find(&conn);
        if (node != buffer_map_.end())
        {
            auto &lifetime = node->second->lifetime();
            if (!lifetime.is_alive())
                throw std::runtime_error("Trying to free a released buffer");
            else
                lifetime.used_count--;
        }
    }

    void grow_age()
    {
        cnt_age_++;
        for (auto &b : buffers_)
        {
            auto &lifetime = b.lifetime();
            if (lifetime.is_alive())
                lifetime.age++;
        }
    }

private:
    size_t next_buffer_id_ = 0;
    size_t cnt_age_ = 0;
    std::list<logical_buffer> &buffers_;
    std::unordered_map<const ir::output_connector *, logical_buffer *> &buffer_map_;
    bool skip_buffer_alias_;
};
}

void schedule_context::generate_compute_sequence()
{
    std::unordered_set<node *> used_inputs;
    auto alloc_visitor = make_relay_ir_visitor([&](node &node)
        {
            if (node.runtime_opcode() == op_input_node)
                used_inputs.emplace(&node);
            else if (node.attributes() & node_attr_action)
                compute_sequence.emplace_back(&node);
        });

    alloc_visitor.visit(outputs);

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
    lifetime_recorder lr(logical_buffers, logical_buffer_map, skip_buffer_alias);
    auto alloc_visitor = make_relay_ir_visitor([&](node &node)
        {
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
    alloc_visitor.visit(outputs);
}

void schedule_context::analyze_buffer_alias()
{
    // bitcast
    {
        pass_manager pmgr(*graph, *this->target);
        pmgr.schedule_context(this);
        pmgr.add_pass<alias_bitcast_buffer_pass>();
        pmgr.run();
    }

    // concat
    {
        pass_manager pmgr(*graph, *this->target);
        pmgr.schedule_context(this);
        pmgr.add_pass<alias_concat_buffer_pass>();
        pmgr.run();
    }
    // 1. add copy to concat
    //{
    //    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
    //        if (auto c = node_cast<concat>(node))
    //        {
    //            auto inputs = c->inputs();
    //            auto outputs = c->output().connections();

    //            // 1. concat by outer-most axis
    //            auto is_simple_concat = (c->axis() == 0 || std::all_of(inputs[0]->shape().begin(), inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
    //            auto &out_buf = logical_buffers.at(&c->output());
    //        }
    //    });
    //    alias_visitor.visit(outputs);
    //}
    //    auto alias_visitor = make_relay_ir_visitor([&](node &node) {
    //        // 1. bitcast
    //        if (auto b = node_cast<bitcast>(node))
    //        {
    //            auto &input = *b->input().connection();
    //            auto &in_buf = logical_buffers.at(&input);
    //            auto &out_buf = logical_buffers.at(&b->output());
    //
    //            // input & rdata should remain locations
    //            if (in_buf.memory_location() == mem_input || in_buf.memory_location() == mem_rdata)
    //            {
    //                // owner is input, parent shape is bitcast's
    //                if (out_buf.memory_location() != mem_output)
    //                {
    //                    shape_t begin(b->output().shape().size(), 0);
    //                    out_buf.parent() = { &in_buf, begin, b->output().shape() };
    //                    out_buf.strides_shape() = b->output().shape();
    //                    b->attributes(b->attributes() & ~node_attr_action);
    //                }
    //            }
    //            else
    //            {
    //                assert(in_buf.memory_location() == mem_data);
    //
    //                // owner transfered to output
    //                shape_t begin(b->output().shape().size(), 0);
    //                in_buf.parent() = { &out_buf, begin, b->output().shape() };
    //                in_buf.strides_shape() = input.shape();
    //                b->attributes(b->attributes() & ~node_attr_action);
    //            }
    //        }
    //        // 2. slice with no_action
    //        else if (auto s = node_cast<slice>(node))
    //        {
    //            if (s->attributes() == node_attr_none)
    //            {
    //                auto &input = *s->input().connection();
    //                auto &in_buf = logical_buffers.at(&input);
    //                auto &out_buf = logical_buffers.at(&s->output());
    //
    //                shape_t begin(s->begin().size(), 0);
    //                for (size_t i = 0; i < s->begin().size(); i++)
    //                    begin[i] = s->begin()[i];
    //                out_buf.parent() = { &in_buf, begin, s->output().shape() };
    //                out_buf.strides_shape() = s->input().shape();
    //            }
    //        }
    //#if 0
    //        // 3. concat
    //        else if (auto c = node_cast<concat>(node))
    //        {
    //            auto inputs = c->inputs();
    //            auto outputs = c->output().connections();
    //            auto child_concats = std::count_if(outputs.begin(), outputs.end(), [](input_connector *in) {
    //                return in->owner().runtime_opcode() == op_concat;
    //            });
    //            auto is_simple_concat = (c->axis() == 0 || std::all_of(inputs[0]->shape().begin(), inputs[0]->shape().begin() + c->axis(), [](size_t dim) { return dim == 1; }));
    //
    //            if (
    //                // input & rdata should be copied to output
    //                std::all_of(inputs.begin(), inputs.end(), [this, is_simple_concat](input_connector *in) {
    //                    auto &in_buf = logical_buffers.at(in->connection());
    //                    return (in_buf.memory_location() != mem_input
    //                               && in_buf.memory_location() != mem_rdata
    //                               && in_buf.memory_location() != mem_output)
    //                        && in->connection()->owner().runtime_opcode() != op_slice
    //                        && (is_simple_concat || in->connection()->owner().runtime_opcode() != op_bitcast);
    //                })
    //                // exclusive concat
    //                && (std::all_of(inputs.begin(), inputs.end(), [](input_connector *in) {
    //                       auto in_outputs = in->connection()->connections();
    //                       auto in_child_concats = std::count_if(in_outputs.begin(), in_outputs.end(), [](input_connector *in) {
    //                           if (auto c = node_cast<concat>(in->owner()))
    //                               return (c->attributes() & node_attr_action) == 0;
    //                           return false;
    //                       });
    //                       return in_child_concats == 0;
    //                   }))
    //                // if any inputs is strided concat, output connections should support strides
    //                && (std::all_of(inputs.begin(), inputs.end(), [this](input_connector *in) {
    //                       if (in->owner().runtime_opcode() == op_concat)
    //                       {
    //                           auto &in_buf = logical_buffers.at(in->connection());
    //                           if (in_buf.no_action_concat_with_strides())
    //                               return false;
    //                       }
    //                       return true;
    //                   })
    //                    || child_concats == 0 || std::all_of(outputs.begin(), outputs.end(), [](input_connector *in) {
    //                           return in->attributes() & cnctr_attr_support_layout_strides;
    //                       })))
    //            {
    //                bool no_action = false;
    //
    //                // no strides support needed
    //                if (is_simple_concat)
    //                {
    //                    no_action = true;
    //                }
    //                else if (std::all_of(inputs.begin(), inputs.end(), [](input_connector *in) { return in->connection()->attributes() & cnctr_attr_support_layout_strides; }))
    //                {
    //                    no_action = true;
    //                    logical_buffers.at(&c->output()).no_action_concat_with_strides() = true;
    //                }
    //
    //                // Fix parent later
    //                if (no_action)
    //                    c->attributes(c->attributes() & ~node_attr_action);
    //            }
    //        }
    //#endif
    //    });
    //    alias_visitor.visit(outputs);
}

void schedule_context::update_offset()
{
    auto visitor = make_relay_ir_visitor<bfs_ir_pre_order_visitor>([&](node &node)
        {
            if (auto b = node_cast<bitcast>(node))
            {
                if (!(b->attributes() & node_attr_action))
                {
                    auto &in_buf = logical_buffer_map.at(b->input().connection());
                    auto &out_buf = logical_buffer_map.at(&b->output());

                    if (in_buf->memory_location() == mem_data && in_buf->parent() && out_buf->parent())
                    {
                        in_buf->parent()->offset += out_buf->parent()->offset;
                    }
                }
            }
            else if (auto c = node_cast<concat>(node))
            {
                if (c->attributes() & node_attr_action)
                    return;

                auto &out_buf = logical_buffer_map.at(&c->output());

                for (auto &in : c->inputs())
                {
                    auto in_buf = logical_buffer_map.at(in->connection());
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
    visitor.visit(outputs);
}

void schedule_context::fix_lifetime()
{
    // Assign parent
    for (auto &bp : logical_buffers)
    {
        auto &p = bp.parent();
        //if (p && p->parent->owner().owner().runtime_opcode() == op_bitcast)
        //{
        //    auto &parent = p->parent->parent();
        //    if (parent && parent->parent->owner().owner().runtime_opcode() == op_concat)
        //    {
        //        p->begin += parent->begin;
        //    }
        //}
        if (p)
        {
            auto parent = p->parent;
            while (parent->parent())
                parent = parent->parent()->parent;
            p->parent = parent;
        }
    }

    // Extend lifetime
    for (auto &bp : logical_buffers)
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
    for (auto &bp : logical_buffers)
    {
        if (!bp.parent())
        {
            auto id = physical_ids.size();
            physical_ids.emplace(&bp, id);
            auto &pb = physical_buffers.emplace_back(id, bp);
            if (auto c = node_cast<constant>(bp.owner().owner()))
                pb.alignment(c->alignment());
        }
    }

    // Assign parents
    for (auto &bp : logical_buffers)
    {
        auto parent = bp.parent() ? bp.parent()->parent : &bp;
        bp.physical() = &physical_buffers.at(physical_ids.at(parent));
    }
}

void schedule_context::allocate_physical_buffers()
{
    allocator_map_t allocators;
    std::vector<std::shared_ptr<buffer_allocator>> allocator_holder;
    target->register_allocators(module_type, allocators, allocator_holder);

    for (auto &usage_p : max_usages)
    {
        // All of rdata live through the module lifetime
        if (usage_p.first == mem_rdata)
        {
            auto it = allocators.find(usage_p.first);
            if (it != allocators.end())
                it->second->base_offset(usage_p.second);
        }
    }

    std::vector<physical_buffer *> orders;
    orders.reserve(physical_buffers.size());
    for (auto &b : physical_buffers)
        orders.emplace_back(&b);
    std::sort(orders.begin(), orders.end(), [](const physical_buffer *lhs, const physical_buffer *rhs)
        { return lhs->lifetime().birth < rhs->lifetime().birth; });

    for (auto &b : orders)
        allocators.at(b->owner().memory_location())->mark(*b);

    for (auto &alloc : allocators)
    {
        alloc.second->finish();
        max_usages.emplace(alloc.first, alloc.second->max_usage());
    }

    for (auto &b : physical_buffers)
        b.allocation() = memory_span { allocators.at(b.owner().memory_location())->allocations().at(&b) };
}

void schedule_context::assign_allocations()
{
    allocator_map_t allocators;
    std::vector<std::shared_ptr<buffer_allocator>> allocator_holder;
    target->register_allocators(module_type, allocators, allocator_holder);

    auto alloc_visitor = make_relay_ir_visitor([&](node &node)
        {
            for (auto out : node.outputs())
            {
                auto &lbuf = *logical_buffer_map.at(out);
                auto &owner = lbuf.physical()->owner();
                auto &memory = lbuf.physical()->allocation();

                // TODO: take account of subbuffer
                buffer_allocation alloc;
                alloc.memory_location = owner.memory_location();
                alloc.type = lbuf.type();
                alloc.size = allocators.at(alloc.memory_location)->get_size_in_bytes(lbuf);
                alloc.shape = lbuf.shape();
                assert(lbuf.strides_shape().size());
                alloc.strides_shape = lbuf.strides_shape();
                alloc.strides = to_strides(alloc.strides_shape);
                alloc.start = memory.start;
                if (lbuf.parent())
                {
                    alloc.start += lbuf.parent()->offset;
                }

                allocations.emplace(out, alloc);
            }
        });
    alloc_visitor.visit(outputs);
}

schedule_result scheduler::schedule(bool skip_buffer_alias)
{
    auto schedule_module = [&](ir::graph &graph, std::span<ir::output_node *> outputs, module_schedule_result &result)
    {
        schedule_context context;
        context.skip_buffer_alias = skip_buffer_alias;
        context.graph = &graph;
        context.module_type = graph.module_type();
        context.outputs = outputs;
        context.target = &target_;

        context.make_logical_buffers();
        if (!skip_buffer_alias)
            context.analyze_buffer_alias();
        context.update_offset();
        context.fix_lifetime();
        context.generate_compute_sequence();
        context.make_physical_buffers();
        context.allocate_physical_buffers();
        context.assign_allocations();

        if (!dump_dir_.empty())
            dump_schedule(context);
        result = module_schedule_result { context };
    };

    auto reachable_graphs = main_graph_.reachable_graphs();
    schedule_result result;
    result.main_module = &main_graph_;
    result.graph_orders.reserve(reachable_graphs.size());
    for (auto subgraph : reachable_graphs)
    {
        schedule_module(*subgraph, subgraph->outputs(), result.modules[subgraph]);
        result.graph_orders.emplace_back(subgraph);
    }

    return result;
}

void scheduler::config_dump(std::filesystem::path dump_dir)
{
    dump_dir_ = std::move(dump_dir);
}

void scheduler::dump_schedule(const schedule_context &context)
{
    std::ofstream writer(dump_dir_ / (context.graph->escaped_name() + ".sched"));

    auto fmt_shape = [&](const logical_buffer &buf)
    {
        auto alloc = context.allocations.at(&buf.owner());
        return fmt::format("<{} {} {} bytes of {}>",
            datatype_names(buf.type()),
            ir::to_string(buf.shape()),
            alloc.size,
            ir::to_string(buf.strides_shape()));
    };

    // 1. allocation
    writer << ".physical_buffer" << std::endl;
    for (auto &buf : context.physical_buffers)
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
    writer << "fn " << context.graph->escaped_name() << "(";

    // 2.2 inputs
    {
        bool comma = false;
        for (auto in : context.graph->inputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *context.logical_buffer_map.at(&in->output());
            auto &pbuf = *lbuf.physical();
            writer << '%' << pbuf.id();
        }
    }

    writer << ") : (";

    // 2.2 input shapes
    {
        bool comma = false;
        for (auto in : context.graph->inputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *context.logical_buffer_map.at(&in->output());
            writer << fmt_shape(lbuf);
        }
    }

    writer << ") -> (";
    // 2.3 output shapes
    {
        bool comma = false;
        for (auto in : context.graph->outputs())
        {
            if (!comma)
                comma = true;
            else
                writer << ", ";

            auto &lbuf = *context.logical_buffer_map.at(in->input().connection());
            writer << fmt_shape(lbuf);
        }
    }

    writer << ")" << std::endl;

#define IDENT "    "

    // 2.4 body
    writer << '{' << std::endl;

    for (auto node : context.compute_sequence)
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

                auto &lbuf = *context.logical_buffer_map.at(out);
                auto &pbuf = *lbuf.physical();
                auto &alloc = context.allocations.at(out);
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

                auto &lbuf = *context.logical_buffer_map.at(in->connection());
                auto &pbuf = *lbuf.physical();
                auto &alloc = context.allocations.at(in->connection());
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

                auto &lbuf = *context.logical_buffer_map.at(in->connection());
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

                auto &lbuf = *context.logical_buffer_map.at(out);
                writer << fmt_shape(lbuf);
            }
        }

        writer << ")" << std::endl;
    }

    writer << '}' << std::endl;
}
