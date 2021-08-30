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
#include <nncase/codegen/model_builder.h>
#include <nncase/ir/op_utils.h>
#include <nncase/runtime/model.h>
#include <nncase/targets/target.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;

model_builder::model_builder(target &target, const schedule::model_schedule_result &sched)
    : target_(target), sched_(sched), dump_asm_(false)
{
}

void model_builder::config_dump(const std::filesystem::path &dump_dir, bool dump_asm)
{
    dump_dir_ = dump_dir;
    dump_asm_ = dump_asm;
}

build_model_result model_builder::build(std::ostream &output)
{
    binary_writer writer(output);
    auto begin_pos = writer.position();

    model_header header {};
    header.identifier = MODEL_IDENTIFIER;
    header.version = MODEL_VERSION;
    header.header_size = sizeof(header);
    header.flags = 0;
    header.alignment = 8;
    header.modules = (uint32_t)sched_.modules.size();

    // Skip model header
    auto header_pos = writer.position();
    writer.skip(sizeof(header));

    for (auto &mod_sched : sched_.modules)
    {
        module_builder_params params { sched_, mod_sched };
        auto builder = target_.create_module_builder(mod_sched.type, mod_sched.type.data(), params);
        builder->config_dump(dump_dir_ / mod_sched.type.data(), dump_asm_);
        builder->build(writer);
        header.alignment = std::max(header.alignment, builder->alignment());
    }

    // Entry point
    for (size_t i = 0; i < sched_.modules.size(); i++)
    {
        auto &mod_sched = sched_.modules[i];
        for (size_t j = 0; j < mod_sched.functions.size(); j++)
        {
            if (sched_.entry_function == &mod_sched.functions[j])
            {
                header.entry_module = (uint32_t)i;
                header.entry_function = (uint32_t)j;
            }
        }
    }

    auto end_pos = writer.position();
    // header
    writer.position(header_pos);
    writer.write(header);
    writer.position(end_pos);

    build_model_result result;
    result.model_size = (size_t)(end_pos - begin_pos);
    return result;
}

size_t model_builder::max_usage(memory_location_t location) const
{
    size_t usage = 0;

    if (location == mem_input)
    {
        // Only take into account of main function's inputs
        auto graph = sched_.entry_function->graph;
        auto &entry_in_allocs = sched_.entry_function->module->allocations;
        for (auto in : graph->inputs())
            usage += entry_in_allocs.at(&in->output()).size;
    }
    else if (location == mem_output)
    {
        // Only take into account of main function's outputs
        auto graph = sched_.entry_function->graph;
        auto &entry_out_allocs = sched_.entry_function->module->allocations;
        for (auto out : graph->outputs())
            usage += entry_out_allocs.at(out->input().connection()).size;
    }
    else if (location != mem_shared_data)
    {
        for (auto &mod : sched_.modules)
        {
            auto it = mod.max_usages.find(location);
            if (it != mod.max_usages.end())
                usage += it->second;
        }
    }
    else
    {
        for (auto &mod : sched_.modules)
        {
            for (auto &shared : mod.shared_max_usages)
                usage += shared.second;
        }
    }

    return usage;
}
