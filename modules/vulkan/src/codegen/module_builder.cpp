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
#include "module_builder.h"

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::vulkan;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

std::unique_ptr<module_builder> codegen::create_vulkan_module_builder(std::string_view module_name, const module_builder_params &params)
{
    return std::make_unique<vulkan_module_builder>(module_name, params);
}

vulkan_module_builder::vulkan_module_builder(std::string_view module_name, const module_builder_params &params)
    : module_builder(256, module_name, params)
{
}

module_type_t vulkan_module_builder::module_type() const noexcept
{
    return vulkan_module_type;
}

section_writer &vulkan_module_builder::text_writer()
{
    return writer(".text");
}

void vulkan_module_builder::emit(ir::node &node)
{
#define DEFINE_OP(op)                              \
    if (node.runtime_opcode() == ir::op::opcode()) \
        return emit(static_cast<ir::op &>(node));
#include "ops.def"
#undef DEFINE_OP
    module_builder::emit(node);
}
