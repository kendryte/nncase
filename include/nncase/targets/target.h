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
#include <memory>
#include <nncase/ir/quantizer.h>
#include <nncase/runtime/model.h>
#include <nncase/schedule/buffer_allocator.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nncase::ir
{
struct node_opcode;
class quantizer;
}

namespace nncase::codegen
{
struct module_builder_params;
class module_builder;
}

namespace nncase::ir::transforms
{
class pass;
class transform_pass;
class pass_manager;
}

namespace nncase
{
struct target_options
{
    virtual ~target_options() = default;

    std::string input_type;
    std::string inference_type;
    float weights_quantize_threshold;
    uint32_t output_quantize_threshold;
    bool quantize_binary;
    bool is_fpga;
};

struct target_attributes
{
};

class NNCASE_API target
{
public:
    virtual ~target() = default;

    target_options &options();

    target_attributes attributes()
    {
        target_attributes attrs {};
        config_attributes(attrs);
        return attrs;
    }

    virtual void register_allocators(const module_type_t &type, schedule::allocator_map_t &allocators, std::vector<std::shared_ptr<schedule::buffer_allocator>> &allocator_holders) = 0;
    virtual void register_evaluator_ops() = 0;
    virtual void register_target_independent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr) = 0;
    virtual void register_target_dependent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr, bool use_ptq) = 0;
    virtual void register_quantize_annotation_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr);
    virtual std::unique_ptr<ir::quantizer> create_quantizer(const module_type_t &type, ir::calibrate_method calib_method);
    virtual void register_quantize_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr, datatype_t quant_type, std::string_view w_quant_type, bool use_mse_quant_w, datatype_t output_type);
    virtual void register_target_dependent_after_quantization_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr);
    virtual void register_target_dependent_after_buffer_fusion_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr);
    virtual void register_allocation_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr) = 0;
    virtual std::unique_ptr<codegen::module_builder> create_module_builder(const module_type_t &type, std::string_view module_name, const codegen::module_builder_params &params);
    virtual void add_quantization_broadcast(std::unordered_set<ir::node_opcode> &opcodes) = 0;

protected:
    virtual std::unique_ptr<target_options> on_create_options() = 0;
    virtual void config_attributes(target_attributes &attrs);

private:
    std::unique_ptr<target_options> options_;
};
}
