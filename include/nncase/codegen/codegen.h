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
#pragma once
#include "codegen_types.h"
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <nncase/ir/graph.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/schedule/scheduler.h>
#include <nncase/targets/target.h>
#include <sstream>
#include <unordered_set>

namespace nncase::codegen
{
class generator;
using emitter_t = std::function<void(ir::node &node, generator &context)>;
using decompiler_t = std::function<void(std::span<const uint8_t> input, std::span<const symbol> symbols, std::ostream &output)>;
using end_emitter_t = std::function<void(generator &context)>;

class generator
{
private:
    struct section
    {
        std::stringstream stream;
        codegen_writer writer;
        std::vector<uint8_t> body;

        section(std::in_place_t = std::in_place)
            : writer(stream)
        {
        }
    };

public:
    generator(nncase::target &target, const schedule::schedule_result &sched, const std::filesystem::path &dump_dir, bool dump_asm = false);

    void gencode(std::ostream &output);

    NNCASE_API const schedule::buffer_allocation &allocation(ir::output_connector &conn) const { return sched_.allocations.at(&conn); }
    NNCASE_API const schedule::buffer_allocation &allocation(ir::input_connector &conn) const { return allocation(*conn.connection()); }
    NNCASE_API size_t max_usage(memory_location_t location) const;
    NNCASE_API codegen_writer &writer(std::string_view section_name);

private:
    void generate_runtime_ops();
    void compile();
    void decompile(std::string_view stage, std::string_view section_name, std::span<const uint8_t> input, std::span<const symbol> symbols);
    void link();
    void write_binary(std::ostream &output);

private:
    nncase::target &target_;
    const schedule::schedule_result &sched_;
    std::filesystem::path dump_dir_;
    bool dump_asm_;
    std::map<std::string, section, std::less<>> section_writer_;
    std::vector<nncase::ir::node *> runtime_ops_;
    std::unordered_set<std::string_view> linked_to_rdata_secs_;

    std::vector<memory_range> inputs_;
    std::vector<runtime_shape_t> input_shapes_;
    std::vector<memory_range> outputs_;
    std::vector<runtime_shape_t> output_shapes_;
    std::vector<std::byte> constants_;
};

NNCASE_API void register_emitter(ir::node_opcode opcode, emitter_t emitter);
NNCASE_API void disable_emitter(ir::node_opcode opcode);
NNCASE_API void register_decompiler(std::string_view section_name, decompiler_t decompiler);
NNCASE_API void register_end_emitter(std::string_view section_name, end_emitter_t emitter);

template <class TNode>
void register_emitter(std::function<void(TNode &node, generator &context)> emitter)
{
    register_emitter(TNode::opcode(), [emitter = std::move(emitter)](ir::node &node, generator &context) {
        emitter(static_cast<TNode &>(node), context);
    });
}
}
