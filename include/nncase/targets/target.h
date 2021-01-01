/* Copyright 2019-2020 Canaan Inc.
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
#include <nncase/runtime/model.h>
#include <nncase/schedule/buffer_allocator.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nncase::ir::transforms
{
class pass;
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
};

class NNCASE_API target
{
public:
    virtual ~target() = default;

    target_options &options();
    virtual runtime::model_target_t model_target() const noexcept = 0;

    virtual void register_allocators(schedule::allocator_map_t &allocators, std::vector<std::unique_ptr<schedule::buffer_allocator>> &allocator_holders) = 0;
    virtual void register_codegen_ops() = 0;
    virtual void register_evaluator_ops() = 0;
    virtual void register_target_independent_passes(ir::transforms::pass_manager &pass_mgr) = 0;
    virtual void register_target_dependent_passes(ir::transforms::pass_manager &pass_mgr) = 0;
    //virtual void add_quantization_checkpoints(ir::transforms::pass_manager &pass_mgr) = 0;
    //virtual void optimize_quantize(ir::quantizer &quantizer, ir::transforms::pass_manager &pass_mgr) = 0;
    virtual void register_allocation_passes(ir::transforms::pass_manager &pass_mgr) = 0;

protected:
    virtual std::unique_ptr<target_options> on_create_options() = 0;

private:
    std::unique_ptr<target_options> options_;
};
}
