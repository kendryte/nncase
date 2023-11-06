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
#include <fstream>
#include <nncase/codegen/module_builder.h>
#include <nncase/io_utils.h>
#include <nncase/ir/debug.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/runtime_type_utils.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/bitio.h>
#include <nncase/runtime/model.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;

namespace
{
std::unordered_set<node_opcode> non_runtime_opcodes { op_input_node, op_output_node, op_uninitialized, op_ignore_node, op_constant };
}

module_builder::module_builder(uint32_t alignment, std::string_view module_name, const module_builder_params &params)
    : dump_asm_(false), alignment_(alignment), module_name_(module_name), params_(params)
{
}

void module_builder::config_dump(const std::filesystem::path &dump_dir, bool dump_asm)
{
    dump_dir_ = dump_dir;
    dump_asm_ = dump_asm;
    if (dump_asm_)
    {
        std::filesystem::create_directories(dump_dir_);
    }
}

const schedule::buffer_allocation &module_builder::allocation(ir::output_connector &conn) const
{
    return params_.module_sched.allocations.at(&conn);
}

section_writer &module_builder::writer(std::string_view section_name)
{
    auto it = section_writer_.find(section_name);
    if (it == section_writer_.end())
        it = section_writer_.emplace(section_name, std::in_place).first;
    return it->second.writer;
}

module_builder::section *module_builder::find_section(std::string_view section_name)
{
    auto it = section_writer_.find(section_name);
    if (it == section_writer_.end())
        return nullptr;
    return &it->second;
}

std::vector<nncase::ir::node *> module_builder::generate_current_runtime_ops()
{
    std::vector<nncase::ir::node *> runtime_ops;
    for (auto &&node : current_function_->compute_sequence)
    {
        if (!non_runtime_opcodes.contains(node->runtime_opcode()))
            runtime_ops.emplace_back(node);
    }

    if (dump_asm_)
    {
        auto func_dump_dir = dump_dir_ / current_function_->graph->escaped_name();
        std::filesystem::create_directories(func_dump_dir);
        std::ofstream file(func_dump_dir / "runtime_ops.txt");
        for (auto node : runtime_ops)
            file << "[" << node->runtime_opcode().name << "] "
                 << node->name() << std::endl;
    }

    return runtime_ops;
}

size_t module_builder::max_usage(memory_location_t location) const
{
    auto it = params_.module_sched.max_usages.find(location);
    if (it != params_.module_sched.max_usages.end())
        return it->second;
    return 0;
}

void module_builder::emit(ir::node &node)
{
    throw std::runtime_error("Emitter for " + node.name() + "[" + std::string(node.runtime_opcode().name) + "] is not found in module " + module_name_ + "[" + module_type().data() + "]");
}

void module_builder::write_constants()
{
    auto it = params_.module_sched.max_usages.find(mem_rdata);
    if (it != params_.module_sched.max_usages.end())
    {
        auto constants = std::make_unique<std::byte[]>(it->second);
        for (auto &func_sched : params_.module_sched.functions)
        {
            for (auto &&node : func_sched.compute_sequence)
            {
                if (auto con = node_cast<constant>(*node))
                {
                    if (con->output().memory_location() == mem_rdata)
                    {
                        auto &alloc = allocation(con->output());
                        auto data = con->data();
                        std::memcpy(constants.get() + alloc.start, data.data(), data.size_bytes());
                    }
                }
            }
        }

        writer(".rdata").write_array(std::span<std::byte const>(constants.get(), it->second));
    }
}

void module_builder::compile()
{
    write_constants();

    begin_emit_module();

    for (auto &func_sched : params_.module_sched.functions)
    {
        current_function_ = &func_sched;

        auto runtime_ops = generate_current_runtime_ops();
        begin_emit_function(func_sched);
        for (auto node : runtime_ops)
            emit(*node);
        end_emit_function(func_sched);

        if (!entry_points_.contains(current_function_))
            throw std::runtime_error("Entry point for " + func_sched.graph->name() + " is not set");
    }

    end_emit_module();

    if (dump_asm_)
    {
        for (auto &section : section_writer_)
            section.second.body = read_stream(section.second.stream);

        for (auto &section : section_writer_)
            decompile("compile", section.first, section.second.body, section.second.writer.symbols());
    }
}

void module_builder::merge_to_rdata_section(std::string_view from)
{
    rdata_section_merges_.emplace(from, std::in_place);
}

function_call_id module_builder::function_id(ir::graph *graph)
{
    for (size_t i = 0; i < params_.model_sched.modules.size(); i++)
    {
        auto &mod_sched = params_.model_sched.modules[i];
        auto func_it = mod_sched.functions_map.find(graph);
        if (func_it != mod_sched.functions_map.end())
        {
            auto func_sched = func_it->second;
            auto &orders = mod_sched.functions;
            if (func_sched >= orders.data() && func_sched < orders.data() + orders.size())
                return { i, (size_t)(func_sched - orders.data()) };
        }
    }

    throw std::invalid_argument("Can't find graph " + graph->name() + " in modules");
}

std::streampos module_builder::get_current_entry_point()
{
    return entry_points_.at(current_function_);
}

void module_builder::set_current_entry_point(std::streampos pos)
{
    entry_points_[current_function_] = pos;
}

void module_builder::set_current_function_text_end(std::streampos pos)
{
    function_text_end_[current_function_] = pos;
}

std::unique_ptr<section_decompiler> module_builder::create_decompiler([[maybe_unused]] std::string_view section_name)
{
    return nullptr;
}

void module_builder::decompile(std::string_view stage, std::string_view section_name, std::span<const uint8_t> input, std::span<const symbol> symbols)
{
    if (auto decompiler = create_decompiler(section_name))
    {
        std::ofstream file(dump_dir_ / (std::string(stage) + std::string(section_name) + ".asm"));
        decompiler->decompile(input, symbols, file);
    }
    else
    {
        std::cout << "WARN: Cannot find a decompiler for section " << section_name << std::endl;
    }
}

void module_builder::generate_merge_info()
{
    if (!rdata_section_merges_.empty())
    {
        auto &rdata_writer = writer(".rdata");

        for (auto &merge_p : rdata_section_merges_)
        {
            auto it = section_writer_.find(merge_p.first);
            if (it != section_writer_.end())
            {
                auto &sec_writer = it->second;
                rdata_writer.align_position(alignment_);
                merge_p.second.start = (uint32_t)rdata_writer.position();
                rdata_writer.write_array<uint8_t>(read_stream(sec_writer.stream));
                merge_p.second.size = (uint32_t)rdata_writer.position() - merge_p.second.start;
            }
        }
    }

    for (auto &section : section_writer_)
    {
        if (!rdata_section_merges_.contains(section.first))
            section.second.body = read_stream(section.second.stream);
    }

    if (dump_asm_)
    {
        std::ofstream file(dump_dir_ / "section-merge.txt");
        for (auto &off : rdata_section_merges_)
            file << off.first << " = " << off.second.start << "@.rdata" << std::endl;
    }
}

void module_builder::generate_symbol_offsets()
{
    for (auto &section : section_writer_)
    {
        if (rdata_section_merges_.contains(section.first))
        {
            auto section_start = rdata_section_merges_.at(section.first).start;
            for (auto &symbol : section.second.writer.symbols())
                symbol_offsets_.emplace(symbol.name, std::make_pair(section_start + (size_t)symbol.streampos, ".rdata"));
        }
        else
        {
            for (auto &symbol : section.second.writer.symbols())
                symbol_offsets_.emplace(symbol.name, std::make_pair((size_t)symbol.streampos, section.first));
        }
    }

    if (dump_asm_)
    {
        std::ofstream file(dump_dir_ / "symbol-addr.txt");
        for (auto &off : symbol_offsets_)
            file << off.first << " = " << off.second.first << "@" << off.second.second << std::endl;
    }
}

void module_builder::write_symbol_refs()
{
    auto rdata_writer = section_writer_.find(".rdata");
    for (auto &section : section_writer_)
    {
        for (auto &ref : section.second.writer.symbol_refs())
        {
            std::span<uint8_t> src_span;
            auto merge_it = rdata_section_merges_.find(section.first);
            if (merge_it == rdata_section_merges_.end())
                src_span = section.second.body;
            else
                src_span = std::span<uint8_t>(rdata_writer->second.body).subspan(merge_it->second.start, merge_it->second.size);

            auto subspan = std::span<uint8_t>(src_span).subspan(ref.streampos);
            bitwriter bw(subspan, ref.bitoffset);
            auto value = symbol_offsets_.at(ref.name).first;
            bw.write(reinterpret_cast<const uint8_t *>(&value), ref.length);
        }
    }
}

void module_builder::link()
{
    generate_merge_info();
    generate_symbol_offsets();
    write_symbol_refs();

    if (dump_asm_)
    {
        for (auto &section : section_writer_)
        {
            if (rdata_section_merges_.contains(section.first))
                decompile("link", section.first, section.second.body, section.second.writer.symbols());
        }
    }
}

void module_builder::write_binary(binary_writer &writer)
{
    // Skip module header
    auto header_pos = writer.position();
    writer.skip(sizeof(module_header));

    // mempools
    for (auto &mem : params_.module_sched.max_usages)
    {
        mempool_desc desc {};
        desc.location = mem.first;
        desc.size = mem.second;
        writer.write(desc);
    }

    // functions
    for (auto &func_sched : params_.module_sched.functions)
        write_function_binary(writer, func_sched);

    // sections
    for (auto &section : section_writer_)
    {
        section_header header {};
        strncpy(header.name, section.first.c_str(), std::size(header.name) - 1);

        auto merge_it = rdata_section_merges_.find(section.first);
        if (merge_it == rdata_section_merges_.end())
        {
            header.flags = 0;
            header.body_start = 0;
            header.body_size = (uint32_t)section.second.body.size();
        }
        else
        {
            header.flags = SECTION_MERGED_INTO_RDATA;
            header.body_start = merge_it->second.start;
            header.body_size = merge_it->second.size;
        }

        // Skip section header
        auto header_pos = writer.position();
        writer.skip(sizeof(header));

        if (merge_it == rdata_section_merges_.end())
        {
            header.body_start = (uint32_t)writer.align_position(alignment_);
            // write content
            writer.write_array(std::span<uint8_t const>(section.second.body));
        }

        // write section header
        auto end_pos = writer.position();
        writer.position(header_pos);
        writer.write(header);
        writer.position(end_pos);
    }

    writer.align_position(8);
    auto end_pos = writer.position();

    // header
    module_header header {};
    header.type = module_type();
    header.version = module_version();
    header.header_size = sizeof(header);
    header.size = (uint32_t)(end_pos - header_pos);
    header.mempools = (uint32_t)params_.module_sched.max_usages.size();
    header.shared_mempools = (uint32_t)params_.module_sched.shared_max_usages.size();
    header.functions = (uint32_t)params_.module_sched.functions.size();
    header.sections = (uint32_t)section_writer_.size();
    header.reserved0 = 0;
    writer.position(header_pos);
    writer.write(header);

    writer.position(end_pos);
}

void module_builder::write_function_binary(binary_writer &writer, const schedule::function_schedule_result &function_sched)
{
    auto write_shape = [&](const shape_t &shape) {
        writer.write((uint32_t)shape.size());
        for (auto dim : shape)
            writer.write((uint32_t)dim);
    };

    std::vector<memory_range> inputs;
    std::vector<shape_t> input_shapes;
    std::vector<memory_range> outputs;
    std::vector<shape_t> output_shapes;

    for (auto &&node : function_sched.compute_sequence)
    {
        if (auto in = node_cast<input_node>(*node))
        {
            auto &alloc = allocation(in->output());
            inputs.emplace_back(alloc.runtime_type());
            input_shapes.emplace_back(alloc.shape);
        }
        else if (auto out = node_cast<output_node>(*node))
        {
            auto &alloc = allocation(out->input());
            outputs.emplace_back(alloc.runtime_type());
            output_shapes.emplace_back(alloc.shape);
        }
    }

    // Skip function header
    auto header_pos = writer.position();
    writer.skip(sizeof(function_header));

    // inputs
    writer.write_array<memory_range>(inputs);
    for (auto &shape : input_shapes)
        write_shape(shape);

    // outputs
    writer.write_array<memory_range>(outputs);
    for (auto &shape : output_shapes)
        write_shape(shape);

    writer.align_position(8);
    auto end_pos = writer.position();

    // header
    function_header header {};
    header.header_size = sizeof(header);
    header.size = (uint32_t)(end_pos - header_pos);
    header.input_pool_size = (uint32_t)function_sched.input_pool_size;
    header.output_pool_size = (uint32_t)function_sched.output_pool_size;
    header.inputs = (uint32_t)inputs.size();
    header.outputs = (uint32_t)outputs.size();
    auto entrypoint = entry_points_.at(&function_sched);
    header.entrypoint = (uint32_t)entrypoint;
    header.text_size = (uint32_t)(function_text_end_.at(&function_sched) - entrypoint);
    writer.position(header_pos);
    writer.write(header);

    writer.position(end_pos);
}

void module_builder::build(binary_writer &writer)
{
    compile();
    link();
    write_binary(writer);
}

void module_builder::begin_emit_function([[maybe_unused]] const schedule::function_schedule_result &function)
{
}

void module_builder::end_emit_function([[maybe_unused]] const schedule::function_schedule_result &function)
{
}

void module_builder::begin_emit_module()
{
}

void module_builder::end_emit_module()
{
}
