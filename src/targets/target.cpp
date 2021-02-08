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
#include <nncase/codegen/stackvm/module_builder.h>
#include <nncase/ir/quantizer.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <nncase/targets/target.h>

using namespace nncase;

target_options &target::options()
{
    if (!options_)
        options_ = on_create_options();
    return *options_;
}

void target::config_attributes(target_attributes &attrs)
{
}

void target::register_quantize_annotation_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

void target::register_quantize_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::quantizer &quantizer, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

std::unique_ptr<ir::quantizer> target::create_quantizer([[maybe_unused]] const module_type_t &type)
{
    return std::make_unique<ir::quantizer>(ir::calibrate_method::no_clip, 1024);
}

std::unique_ptr<codegen::module_builder> target::create_module_builder(const module_type_t &type, std::string_view module_name, const schedule::module_schedule_result &sched)
{
    if (type == runtime::stackvm::stackvm_module_type)
        return codegen::create_stackvm_module_builder(module_name, sched);
    else
        throw std::runtime_error("Module builder for module " + std::string(module_name) + "[" + type.data() + "] is not found");
}
