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
#include <nncase/ir/ops/constant.h>
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
}






void schedule_context::assign_allocations()
{
    allocator_map_t allocators;
    std::vector<std::shared_ptr<buffer_allocator>> allocator_holder;
    target->register_allocators(module_type, allocators, allocator_holder);

    auto alloc_visitor = make_relay_ir_visitor([&](node &node) {
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
    auto schedule_module = [&](ir::graph &graph, std::span<ir::output_node *> outputs, module_schedule_result &result) {
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

    auto fmt_shape = [&](const logical_buffer &buf) {
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
