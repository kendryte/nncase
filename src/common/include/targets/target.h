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
#include <hlir/quantizer.h>
#include <hlir/transforms/pass.h>
#include <llir/transforms/pass.h>
#include <memory>
#include <scheduler/memory_allocator.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nncase
{
struct target_options
{
    std::string input_type;
    std::string inference_type;
    float weights_quantize_threshold;
    uint32_t output_quantize_threshold;
    bool quantize_binary;
};

class target
{
public:
    target(const target_options &options)
        : options_(options) {}
    virtual ~target() = default;

    const target_options &options() const noexcept { return options_; }

    virtual void fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<scheduler::memory_allocator>> &allocator_holders) = 0;
    virtual void registry_codegen_ops() = 0;
    virtual void registry_evaluator_ops() = 0;
    virtual void optimize_target_independent(hlir::transforms::pass_manager &pass_mgr) = 0;
    virtual void optimize_target_dependent(hlir::transforms::pass_manager &pass_mgr) = 0;
    virtual void add_quantization_checkpoints(hlir::transforms::pass_manager &pass_mgr) = 0;
    virtual void optimize_quantize(hlir::quantizer &quantizer, hlir::transforms::pass_manager &pass_mgr) = 0;
    virtual void add_quantization_broadcast(std::unordered_set<hlir::node_opcode> &opcodes) = 0;
    virtual void optimize_llir(llir::transforms::pass_manager &pass_mgr) = 0;

protected:
    target_options options_;
};
}
