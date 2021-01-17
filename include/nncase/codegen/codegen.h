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
    module_builder(const std::filesystem::path &dump_dir, bool dump_asm = false);
    module_builder(module_builder &) = delete;
    module_builder(module_builder &&) = delete;

    void gencode(std::ostream &output);

    const schedule::buffer_allocation &allocation(ir::output_connector &conn) const;
    const schedule::buffer_allocation &allocation(ir::input_connector &conn) const { return allocation(*conn.connection()); }
    section_writer &writer(std::string_view section_name);

    virtual module_type_t module_type() const noexcept = 0;
    virtual std::unique_ptr<module_decompiler> create_decompiler(std::string_view section_name) = 0;
    void compile_function(std::string_view function_name, const schedule::schedule_result &sched);

protected:
    void merge_section(std::string_view from, std::string_view to);

    virtual void begin_emit() { }
    virtual void emit(ir::node &node) = 0;
    virtual void end_emit() { }

private:
    std::vector<nncase::ir::node *> generate_runtime_ops(const schedule::schedule_result &sched);
    void decompile(std::string_view stage, std::string_view section_name, std::span<const uint8_t> input, std::span<const symbol> symbols);
    void link();
    void write_binary(std::ostream &output);

private:
    std::filesystem::path dump_dir_;
    bool dump_asm_;
    std::map<std::string, section, std::less<>> section_writer_;
    std::map<std::string, std::unordered_set<std::string>, std::less<>> section_merges_;
    const schedule::schedule_result *cnt_sched_;
};
}
