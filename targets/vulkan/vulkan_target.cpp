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
#include "vulkan_target.h"
#include <nncase/codegen/vulkan/module_builder.h>
#include <nncase/plugin_loader.h>

#if defined(_MSC_VER)
#define VULKAN_TARGET_API __declspec(dllexport)
#else
#define VULKAN_TARGET_API
#endif

using namespace nncase;
using namespace nncase::targets;
using namespace nncase::runtime;

extern "C"
{
    VULKAN_TARGET_API target *create_target()
    {
        return new vulkan_target();
    }
}

std::unique_ptr<codegen::module_builder> vulkan_target::create_module_builder(const module_type_t &type, std::string_view module_name, const codegen::module_builder_params &params)
{
    if (type == runtime::vulkan::vulkan_module_type)
        return codegen::create_vulkan_module_builder(module_name, params);
    return neutral_target::create_module_builder(type, module_name, params);
}
