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
#include "../cpu/target.h"
#include <targets/target.h>

namespace nncase
{
class k210_target : public cpu_target
{
public:
    using cpu_target::cpu_target;

    void fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<scheduler::memory_allocator>> &allocator_holders) override;
    void registry_codegen_ops() override;
    void registry_evaluator_ops() override;
    void optimize_target_independent(hlir::transforms::pass_manager &pass_mgr) override;
    void optimize_target_dependent(hlir::transforms::pass_manager &pass_mgr) override;
    void add_quantization_checkpoints(hlir::transforms::pass_manager &pass_mgr) override;
    void optimize_quantize(hlir::quantizer &quantizer, hlir::transforms::pass_manager &pass_mgr) override;
};
}
