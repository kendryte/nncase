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
#include <nncase/targets/target.h>

namespace nncase::targets
{
class NNCASE_API neutral_target : public target
{
public:
    using target::target;

    void register_allocators(const module_type_t &type, schedule::allocator_map_t &allocators, std::vector<std::unique_ptr<schedule::buffer_allocator>> &allocator_holders) override;
    void register_evaluator_ops() override;
    void register_target_independent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr) override;
    void register_target_dependent_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr) override;
    void register_allocation_passes(const module_type_t &type, ir::transforms::pass_manager &pass_mgr) override;

protected:
    void move_transpose_transform(ir::transforms::pass &pass, bool add_constant_folding = true);
    void fold_pad_conv_transform(ir::transforms::pass &pass, bool add_constant_folding = true);
    void fold_dilated_conv_transform(ir::transforms::pass &pass, bool add_constant_folding = true);
    void add_default_transforms(ir::transforms::pass &pass, bool add_constant_folding = true);

    std::unique_ptr<target_options> on_create_options() override;
};
}
