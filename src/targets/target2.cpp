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
#include <nncase/codegen/stackvm/module_builder.h>
#include <nncase/ir/quantizer.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <nncase/targets/target.h>
#include <nncase/transforms/pass.h>

using namespace nncase;

target_options &target::options()
{
    if (!options_)
        options_ = on_create_options();
    return *options_;
}

void target::config_attributes([[maybe_unused]] target_attributes &attrs)
{
}

void target::register_quantize_annotation_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

void target::register_quantize_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr, [[maybe_unused]] datatype_t quant_type, [[maybe_unused]] datatype_t w_quant_type)
{
}

void target::add_quantization_broadcast([[maybe_unused]] std::unordered_set<ir::node_opcode> &opcodes)
{
}

void target::register_target_dependent_after_quantization_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

void target::register_target_dependent_after_buffer_fusion_passes([[maybe_unused]] const module_type_t &type, [[maybe_unused]] ir::transforms::pass_manager &pass_mgr)
{
}

std::unique_ptr<ir::quantizer> target::create_quantizer([[maybe_unused]] const module_type_t &type, ir::calibrate_method calib_method)
{
    return std::make_unique<ir::quantizer>(calib_method, 1024);
}

std::unique_ptr<codegen::module_builder> target::create_module_builder(const module_type_t &type, std::string_view module_name, const codegen::module_builder_params &params)
{
    if (type == runtime::stackvm::stackvm_module_type)
        return codegen::create_stackvm_module_builder(module_name, params);
    else
        throw std::runtime_error("Module builder for module " + std::string(module_name) + "[" + type.data() + "] is not found");
}
