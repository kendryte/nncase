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
#include <sstream>
#include <unordered_set>

namespace nncase::codegen
{
class NNCASE_API module_decompiler
{
public:
    virtual void decompile(std::span<const uint8_t> input, std::span<const symbol> symbols, std::ostream &output) = 0;
};

class NNCASE_API module_builder
{
private:
    struct section
    {
        std::stringstream stream;
        section_writer writer;
        std::vector<uint8_t> body;

        section(std::in_place_t = std::in_place)
            : writer(stream)
        {
        }
    };

public:
    module_builder(const schedule::schedule_result &sched, const std::filesystem::path &dump_dir, bool dump_asm = false);

    void gencode(std::ostream &output);

    const schedule::buffer_allocation &allocation(ir::output_connector &conn) const { return sched_.allocations.at(&conn); }
    const schedule::buffer_allocation &allocation(ir::input_connector &conn) const { return allocation(*conn.connection()); }
    size_t max_usage(memory_location_t location) const;
    section_writer &writer(std::string_view section_name);

    virtual module_type_t module_type() const noexcept = 0;
    virtual std::unique_ptr<module_decompiler> create_decompiler(std::string_view section_name) = 0;

protected:
    void add_section_merge(std::string_view from, std::string_view to);

    virtual void emit(ir::node &node) = 0;
    virtual void end_emit() { }

private:
    void generate_runtime_ops();
    void compile();
    void decompile(std::string_view stage, std::string_view section_name, std::span<const uint8_t> input, std::span<const symbol> symbols);
    void link();
    void write_binary(std::ostream &output);

private:
    const schedule::schedule_result &sched_;
    std::filesystem::path dump_dir_;
    bool dump_asm_;
    std::map<std::string, section, std::less<>> section_writer_;
    std::vector<nncase::ir::node *> runtime_ops_;
    std::map<std::string, std::unordered_set<std::string>, std::less<>> section_merges_;

    std::vector<memory_range> inputs_;
    std::vector<runtime_shape_t> input_shapes_;
    std::vector<memory_range> outputs_;
    std::vector<runtime_shape_t> output_shapes_;
    std::vector<std::byte> constants_;
};
}
