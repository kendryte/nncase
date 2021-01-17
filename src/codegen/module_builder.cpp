/* Copyright 2020 Canaan Inc.
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
#include <nncase/codegen/codegen.h>
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

module_builder::module_builder(const std::filesystem::path &dump_dir, bool dump_asm)
    : dump_dir_(dump_dir), dump_asm_(dump_asm), cnt_sched_(nullptr)
{
}

const schedule::buffer_allocation &module_builder::allocation(ir::output_connector &conn) const
{
    return cnt_sched_->allocations.at(&conn);
}

section_writer &module_builder::writer(std::string_view section_name)
{
    auto it = section_writer_.find(section_name);
    if (it == section_writer_.end())
        it = section_writer_.emplace(section_name, std::in_place).first;
    return it->second.writer;
}

std::vector<nncase::ir::node *> module_builder::generate_runtime_ops(const schedule::schedule_result &sched)
{
    std::vector<nncase::ir::node *> runtime_ops;
    for (auto &&node : sched.compute_sequence)
        runtime_ops.emplace_back(node);

    if (dump_asm_)
    {
        std::ofstream file(dump_dir_ / "runtime_ops.txt");
        for (auto node : runtime_ops)
            file << "[" << node->runtime_opcode().name << "] "
                 << node->name() << std::endl;
    }

    return runtime_ops;
}

void module_builder::compile_function(std::string_view function_name, const schedule::schedule_result &sched)
{
    auto runtime_ops = generate_runtime_ops(sched);
    begin_emit();
    for (auto node : runtime_ops)
        emit(*node);
    end_emit();
}

void module_builder::merge_section(std::string_view from, std::string_view to)
{
    section_merges_[std::string(from)].emplace(to);
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

void module_builder::link()
{
    for (auto &section : section_writer_)
        section.second.body = read_stream(section.second.stream);

    if (dump_asm_)
    {
        for (auto &section : section_writer_)
            decompile("compile", section.first, section.second.body, section.second.writer.symbols());
    }
}

void module_builder::write_binary(std::ostream &output)
{
}

//void module_builder::link()
//{
//    // 1. Link non '.text' sections to rdata
//    size_t rdata_start = max_usage(mem_rdata);
//    std::unordered_map<std::string_view, size_t> offsets;
//    for (auto &section : section_writer_)
//    {
//        if (section.first != ".text")
//        {
//            offsets.emplace(section.first, rdata_start);
//            section.second.stream.seekg(0, std::ios::end);
//            rdata_start += (size_t)section.second.stream.tellg();
//            linked_to_rdata_secs_.emplace(section.first);
//        }
//    }
//
//    if (dump_asm_)
//    {
//        std::ofstream file(dump_dir_ / "section-base.txt");
//        for (auto &off : offsets)
//            file << off.first << " = " << off.second << "@.rdata" << std::endl;
//    }
//
//    // 2. Generate symbol offsets
//    std::unordered_map<std::string_view, std::pair<size_t, std::string_view>> symbol_offsets;
//    for (auto &section : section_writer_)
//    {
//        if (section.first != ".text")
//        {
//            auto section_start = offsets.at(section.first);
//            for (auto &symbol : section.second.writer.symbols())
//                symbol_offsets.emplace(symbol.name, std::make_pair(section_start + (size_t)symbol.streampos, ".rdata"));
//        }
//        else
//        {
//            for (auto &symbol : section.second.writer.symbols())
//                symbol_offsets.emplace(symbol.name, std::make_pair((size_t)symbol.streampos, section.first));
//        }
//    }
//
//    if (dump_asm_)
//    {
//        std::ofstream file(dump_dir_ / "symbol-addr.txt");
//        for (auto &off : symbol_offsets)
//            file << off.first << " = " << off.second.first << "@" << off.second.second << std::endl;
//    }
//
//    // 3. Write symbol refs
//    for (auto &section : section_writer_)
//    {
//        for (auto &ref : section.second.writer.symbol_refs())
//        {
//            bitwriter bw(std::span(section.second.body).subspan(ref.streampos), ref.bitoffset);
//            auto value = symbol_offsets.at(ref.name).first;
//            bw.write(reinterpret_cast<const uint8_t *>(&value), ref.length);
//        }
//    }
//
//    if (dump_asm_)
//    {
//        for (auto &section : section_writer_)
//            decompile("link", section.first, section.second.body, section.second.writer.symbols());
//    }
//}

//void module_builder::write_binary(std::ostream &output)
//{
//    constants_.resize(sched_.max_usages.at(mem_rdata));
//
//    for (auto &&node : sched_.compute_sequence)
//    {
//        if (auto in = node_cast<input_node>(*node))
//        {
//            auto &alloc = allocation(in->output());
//            inputs_.emplace_back(alloc.runtime_type());
//            input_shapes_.emplace_back(to(alloc.shape));
//        }
//        else if (auto out = node_cast<output_node>(*node))
//        {
//            auto &alloc = allocation(out->input());
//            outputs_.emplace_back(alloc.runtime_type());
//            output_shapes_.emplace_back(to(alloc.shape));
//        }
//        else if (auto con = node_cast<constant>(*node))
//        {
//            auto &alloc = allocation(con->output());
//            auto data = con->data();
//            std::memcpy(constants_.data() + alloc.start, data.data(), data.size_bytes());
//        }
//    }
//
//    {
//        binary_writer writer(output);
//        // Skip model header
//        auto header_pos = writer.position();
//        writer.skip(sizeof(model_header));
//
//        // inputs
//        writer.write_array<memory_range>(inputs_);
//        writer.write_array<runtime_shape_t>(input_shapes_);
//
//        // outputs
//        writer.write_array<memory_range>(outputs_);
//        writer.write_array<runtime_shape_t>(output_shapes_);
//
//        // Skip sections desc
//        std::unordered_map<std::string_view, section_desc> section_pos;
//        // memories
//        for (auto &mem : sched_.max_usages)
//        {
//            auto name = memory_to_section(mem.first);
//            auto &desc = section_pos[name];
//            std::copy(name.begin(), name.end(), desc.name);
//            desc.size = mem.second;
//            desc.size_in_file = 0;
//            desc.offset = 0;
//        }
//
//        auto section_desc_pos = writer.position();
//        writer.skip(sizeof(section_desc) * (section_writer_.size() + sched_.max_usages.size() - linked_to_rdata_secs_.size()));
//
//        // text
//        writer.align_position(8);
//        {
//            auto &txt_sec = section_writer_.at(".text");
//            auto &desc = section_pos[".text"];
//            strcpy(desc.name, ".text");
//            desc.offset = (uint32_t)writer.relative_offset();
//            desc.size_in_file = txt_sec.body.size();
//            desc.size = desc.size_in_file;
//            writer.write_array<uint8_t>(txt_sec.body);
//        }
//
//        // rdata
//        writer.align_position(8);
//        {
//            auto &rdata_desc = section_pos.at(".rdata");
//            rdata_desc.offset = (uint32_t)writer.relative_offset();
//            rdata_desc.size_in_file = rdata_desc.size;
//            writer.write_array<std::byte>(constants_);
//        }
//
//        if (dump_asm_)
//        {
//            std::ofstream file(dump_dir_ / "link.rdata.bin", std::ios::binary);
//            binary_writer bw(file);
//            bw.write_array<std::byte>(constants_);
//        }
//
//        for (auto &section : section_writer_)
//        {
//            if (section.first != ".text")
//            {
//                auto &desc = section_pos[section.first];
//                std::copy(section.first.begin(), section.first.end(), desc.name);
//                desc.offset = (uint32_t)writer.relative_offset();
//                desc.size_in_file = section.second.body.size();
//                desc.size = desc.size_in_file;
//                writer.write_array<uint8_t>(section.second.body);
//
//                auto &rdata_desc = section_pos.at(".rdata");
//                rdata_desc.size_in_file += desc.size;
//                rdata_desc.size += desc.size;
//
//                if (dump_asm_)
//                {
//                    std::ofstream file(dump_dir_ / ("link" + section.first + ".bin"), std::ios::binary);
//                    binary_writer bw(file);
//                    bw.write_array<uint8_t>(section.second.body);
//                }
//            }
//        }
//
//        auto end_pos = writer.position();
//
//        // header
//        module_builder header;
//        model_header.identifier = MODEL_IDENTIFIER;
//        model_header.checksum = 0;
//        model_header.version = MODEL_VERSION;
//        model_header.flags = 0;
//        model_header.target = target_.model_target();
//        model_header.memories = (uint32_t)sched_.max_usages.size();
//        model_header.sections = (uint32_t)(section_pos.size() - linked_to_rdata_secs_.size());
//        model_header.inputs = (uint32_t)inputs_.size();
//        model_header.outputs = (uint32_t)outputs_.size();
//        writer.position(header_pos);
//        writer.write(model_header);
//
//        // section pos
//        writer.position(section_desc_pos);
//        for (auto &pos : section_pos)
//        {
//            if (!linked_to_rdata_secs_.contains(pos.first))
//                writer.write(pos.second);
//        }
//
//        writer.position(end_pos);
//    }
//
//    // TODO: checksum
//}

void module_builder::gencode(std::ostream &output)
{
    link();
    write_binary(output);
}
