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
#include <nncase/codegen/model_builder.h>
#include <nncase/runtime/model.h>
#include <nncase/targets/target.h>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;

model_builder::model_builder(target &target, const schedule::schedule_result &sched)
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
    header.flags = 0;
    header.alignment = 8;
    header.modules = (uint32_t)sched_.modules.size();

    // Skip module header
    auto header_pos = writer.position();
    writer.skip(sizeof(header));

    uint32_t main_module_id = 0;
    for (auto &graph : sched_.graph_orders)
    {
        auto &mod = sched_.modules.at(graph);
        module_builder_params params { sched_, mod };
        auto builder = target_.create_module_builder(graph->module_type(), graph->name(), params);
        builder->config_dump(dump_dir_, dump_asm_);
        builder->build(writer);
        header.alignment = std::max(header.alignment, builder->alignment());

        if (graph == sched_.main_module)
            header.main_module = main_module_id;
        main_module_id++;
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
    for (auto &mod : sched_.modules)
    {
        auto it = mod.second.max_usages.find(location);
        if (it != mod.second.max_usages.end())
            usage += it->second;
    }

    return usage;
}
