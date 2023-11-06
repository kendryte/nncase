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
class NNCASE_API section_decompiler
{
public:
    virtual ~section_decompiler() = default;
    virtual void decompile(std::span<const uint8_t> input, std::span<const symbol> symbols, std::ostream &output) = 0;
};

struct module_builder_params
{
    const schedule::model_schedule_result &model_sched;
    const schedule::module_schedule_result &module_sched;
};

struct function_call_id
{
    size_t module_id;
    size_t function_id;
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

    struct rdata_merge_info
    {
        size_t start;
        size_t size;

        rdata_merge_info(std::in_place_t = std::in_place)
        {
        }
    };

public:
    module_builder(uint32_t alignment, std::string_view module_name, const module_builder_params &params);
    module_builder(module_builder &) = delete;
    module_builder(module_builder &&) = delete;
    virtual ~module_builder() = default;

    uint32_t alignment() const noexcept { return alignment_; }
    void config_dump(const std::filesystem::path &dump_dir, bool dump_asm);
    void build(binary_writer &writer);

    const schedule::buffer_allocation &allocation(ir::output_connector &conn) const;
    const schedule::buffer_allocation &allocation(ir::input_connector &conn) const { return allocation(*conn.connection()); }
    size_t max_usage(memory_location_t location) const;
    section_writer &writer(std::string_view section_name);

    virtual module_type_t module_type() const noexcept = 0;
    virtual uint32_t module_version() const noexcept = 0;
    virtual std::unique_ptr<section_decompiler> create_decompiler(std::string_view section_name);

protected:
    section *find_section(std::string_view section_name);
    void merge_to_rdata_section(std::string_view from);
    function_call_id function_id(ir::graph *graph);
    std::streampos get_current_entry_point();
    void set_current_entry_point(std::streampos pos);
    void set_current_function_text_end(std::streampos pos);

    virtual void begin_emit_module();
    virtual void begin_emit_function(const schedule::function_schedule_result &function);
    virtual void end_emit_function(const schedule::function_schedule_result &function);
    virtual void emit(ir::node &node);
    virtual void end_emit_module();

protected:
    std::filesystem::path dump_dir_;
    bool dump_asm_;

private:
    std::vector<nncase::ir::node *> generate_current_runtime_ops();
    void compile();
    void decompile(std::string_view stage, std::string_view section_name, std::span<const uint8_t> input, std::span<const symbol> symbols);

    void write_constants();
    void generate_merge_info();
    void generate_symbol_offsets();
    void write_symbol_refs();
    void link();
    void write_binary(binary_writer &writer);
    void write_function_binary(binary_writer &writer, const schedule::function_schedule_result &function_sched);

private:
    uint32_t alignment_;
    std::string module_name_;
    const module_builder_params &params_;
    std::map<std::string, section, std::less<>> section_writer_;
    std::map<std::string, rdata_merge_info, std::less<>> rdata_section_merges_;
    std::unordered_map<std::string_view, std::pair<size_t, std::string_view>> symbol_offsets_;

    const schedule::function_schedule_result *current_function_;
    std::unordered_map<const schedule::function_schedule_result *, std::streampos> entry_points_;
    std::unordered_map<const schedule::function_schedule_result *, std::streampos> function_text_end_;
};
}
