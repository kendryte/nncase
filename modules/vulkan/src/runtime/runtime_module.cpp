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
#include "runtime_module.h"
#include "vulkan_error.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <vulkan/vulkan.hpp>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> vulkan_runtime_module::initialize_core(runtime_module_init_context &context) noexcept
{
    assert(context.is_section_pinned());
    auto data_pool = mempool(mem_data);
    if (data_pool.size)
    {
        data_.reset(new (std::nothrow) gsl::byte[data_pool.size]);
        if (!data_)
            return err(std::errc::not_enough_memory);
    }

    rdata_ = context.section(".rdata");
    text_ = context.section(".text");
    return ok();
}

result<runtime_tensor> vulkan_runtime_module::allocate_input_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(input_desc(index).datatype, input_shape(index));
}

result<runtime_tensor> vulkan_runtime_module::allocate_output_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(output_desc(index).datatype, output_shape(index));
}

result<void> vulkan_runtime_module::validate_input_tensor(NNCASE_UNUSED size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> vulkan_runtime_module::validate_output_tensor(NNCASE_UNUSED size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> vulkan_runtime_module::initialize_vulkan() noexcept
{
    vk::ApplicationInfo app_info("nncase.runtime", 1, "nncase", 1, VK_API_VERSION_1_1);
    vk::InstanceCreateInfo create_info({}, &app_info);
    try_var(instance, vk::to_result(vk::createInstance(create_info)));

    return ok();
}

result<void> vulkan_runtime_module::run_core() noexcept
{
    return ok();
}

result<std::unique_ptr<runtime_module>> vulkan::create_vulkan_runtime_module()
{
    std::unique_ptr<runtime_module> mod(new (std::nothrow) vulkan_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
