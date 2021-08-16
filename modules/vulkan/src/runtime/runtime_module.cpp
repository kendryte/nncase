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
#include "runtime_function.h"
#include "vulkan_error.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

vulkan_runtime_module::~vulkan_runtime_module()
{
    free_vulkan_resources();
}

result<void> vulkan_runtime_module::initialize_before_functions(runtime_module_init_context &context) noexcept
{
    assert(context.is_section_pinned());
    auto descs = context.section(DESCRIPTORS_SECTION_NAME).as_span<const uint32_t>();
    descriptor_sets_ = descs[0];
    descriptors_ = descs[1];
    shader_ = context.section(".shader");

    checked_try(initialize_vulkan());

    // TODO: Load rdata
    //rdata_ = context.section(".rdata");
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan() noexcept
{
    checked_try(initialize_vulkan_instance());
    checked_try(initialize_vulkan_device());
    checked_try(initialize_vulkan_memory());
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_instance() noexcept
{
    vk::ApplicationInfo app_info("nncase.runtime", 1, "nncase", 1, VK_API_VERSION_1_1);
    vk::InstanceCreateInfo create_info({}, &app_info);
    checked_try_set(instance_, vk::to_result(vk::createInstance(create_info)));
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_device() noexcept
{
    checked_try_set(physical_device_, select_physical_device());
    auto queue_families = physical_device_.getQueueFamilyProperties();
    checked_try_set(compute_queue_index_, select_queue_family(queue_families, { vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, vk::QueueFlagBits::eTransfer }));

    float priorities[] = { 0.0f };
    vk::DeviceQueueCreateInfo queue_create_info({}, compute_queue_index_, 1, priorities);
    vk::DeviceCreateInfo device_create_info({}, queue_create_info);
    checked_try_set(device_, vk::to_result(physical_device_.createDevice(device_create_info)));
    compute_queue_ = device_.getQueue(compute_queue_index_, 0);

    vk::DescriptorPoolSize descp_size(vk::DescriptorType::eStorageBuffer, descriptors_);
    vk::DescriptorPoolCreateInfo descp_cinfo({}, descriptor_sets_, descp_size);
    checked_try_set(buffer_desc_pool_, vk::to_result(device_.createDescriptorPool(descp_cinfo)));

    vk::CommandPoolCreateInfo cmdp_cinfo({}, compute_queue_index_);
    try_set(cmd_pool_, vk::to_result(device_.createCommandPool(cmdp_cinfo)));
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_memory() noexcept
{
    auto data_mem = mempool(mem_data);
    if (data_mem.size)
    {
        checked_try_set(data_buffer_, allocate_vulkan_buffer(data_mem.size));
        checked_try_set(data_mem_, allocate_vulkan_memory({ {}, vk::MemoryPropertyFlagBits::eDeviceLocal, {} }, data_buffer_));
        checked_try(bind_vulkan_buffer(data_buffer_, data_mem_));
    }
    return ok();
}

result<vk::DeviceMemory> vulkan_runtime_module::allocate_vulkan_memory(const select_options<vk::MemoryPropertyFlagBits> &options, vk::Buffer buffer) noexcept
{
    auto req = device_.getBufferMemoryRequirements(buffer);
    auto properties = physical_device_.getMemoryProperties();
    checked_try_var(type_index, select_memory_type(properties, options, req.size));
    vk::MemoryAllocateInfo allocate(req.size, static_cast<uint32_t>(type_index));
    return vk::to_result(device_.allocateMemory(allocate));
}

result<vk::Buffer> vulkan_runtime_module::allocate_vulkan_buffer(size_t required_size) noexcept
{
    vk::BufferCreateInfo info({}, (vk::DeviceSize)required_size,
        vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive, 1, &compute_queue_index_);
    return vk::to_result(device_.createBuffer(info));
}

result<void> vulkan_runtime_module::bind_vulkan_buffer(vk::Buffer buffer, vk::DeviceMemory memory) noexcept
{
    return vk::to_result(device_.bindBufferMemory(buffer, memory, 0));
}

result<void> vulkan_runtime_module::add_pipeline(vk::Pipeline pipeline) noexcept
{
    try
    {
        pipelines_owner_.emplace_back(pipeline);
        return ok();
    }
    catch (const std::bad_alloc &)
    {
        return err(std::errc::not_enough_memory);
    }
}

result<vk::PhysicalDevice> vulkan_runtime_module::select_physical_device() noexcept
{
    vk::PhysicalDevice *intergrated = nullptr;

    checked_try_var(devices, vk::to_result(instance_.enumeratePhysicalDevices()));
    for (auto &device : devices)
    {
        auto properties = device.getProperties();
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            return ok(device);
        else if (properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            intergrated = &device;
    }

    if (intergrated)
        return ok(*intergrated);
    else if (!devices.empty())
        return ok(devices.front());
    else
        return err(std::errc::no_such_device);
}

result<uint32_t> vulkan_runtime_module::select_queue_family(const std::vector<vk::QueueFamilyProperties> &families, const select_options<vk::QueueFlagBits> options) noexcept
{
    // 1. try required & preferred & !not_preferred
    for (uint32_t i = 0; i < families.size(); i++)
    {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried
            && (flags & options.preferred) == options.preferred
            && !(flags & options.not_preferred))
            return ok(i);
    }

    // 2. try required & preferred
    for (uint32_t i = 0; i < families.size(); i++)
    {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried
            && (flags & options.preferred) == options.preferred)
            return ok(i);
    }

    // 3. try required
    for (uint32_t i = 0; i < families.size(); i++)
    {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried)
            return ok(i);
    }

    std::cerr << "Cannot find available queue: " << to_string(options.requried) << std::endl;
    return err(std::errc::no_such_device);
}

result<size_t> vulkan_runtime_module::select_memory_type(const vk::PhysicalDeviceMemoryProperties &properties, const select_options<vk::MemoryPropertyFlagBits> &options, size_t required_size) noexcept
{
    auto &memory_types = properties.memoryTypes;
    // 1. try required & preferred & !not_preferred
    for (size_t i = 0; i < memory_types.size(); i++)
    {
        auto &type = memory_types[i];
        auto flags = type.propertyFlags;
        if ((flags & options.requried) == options.requried
            && (flags & options.preferred) == options.preferred
            && !(flags & options.not_preferred)
            && properties.memoryHeaps[type.heapIndex].size >= required_size)
            return ok(i);
    }

    // 2. try required & preferred
    for (size_t i = 0; i < memory_types.size(); i++)
    {
        auto &type = memory_types[i];
        auto flags = type.propertyFlags;
        if ((flags & options.requried) == options.requried
            && (flags & options.preferred) == options.preferred
            && properties.memoryHeaps[type.heapIndex].size >= required_size)
            return ok(i);
    }

    // 3. try required
    for (size_t i = 0; i < memory_types.size(); i++)
    {
        auto &type = memory_types[i];
        auto flags = type.propertyFlags;
        if ((flags & options.requried) == options.requried
            && properties.memoryHeaps[type.heapIndex].size >= required_size)
            return ok(i);
    }

    std::cerr << "Cannot allocate " << std::to_string(required_size)
              << "B memory: " << to_string(options.requried) << std::endl;
    return err(std::errc::not_enough_memory);
}

void vulkan_runtime_module::free_vulkan_resources() noexcept
{
    for (auto &func : functions())
        func.reset();

    device_.destroyCommandPool(cmd_pool_);
    device_.destroyDescriptorPool(buffer_desc_pool_);
    for (auto p : pipelines_owner_)
        device_.destroyPipeline(p);
    device_.destroyBuffer(data_buffer_);
    device_.freeMemory(data_mem_);
    device_.destroy({});
    instance_.destroy({});
}

result<std::unique_ptr<runtime_function>> vulkan_runtime_module::create_function() noexcept
{
    std::unique_ptr<runtime_function> mod(new (std::nothrow) vulkan_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>> vulkan::create_vulkan_runtime_module()
{
    std::unique_ptr<runtime_module> mod(new (std::nothrow) vulkan_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

extern "C"
{
    NNCASE_MODULES_VULKAN_API void RUNTIME_MODULE_ACTIVATOR_NAME(result<std::unique_ptr<runtime_module>> &result)
    {
        result = create_vulkan_runtime_module();
    }
}

#ifndef NNCASE_SIMULATOR
runtime_registration nncase::runtime::builtin_runtimes[] = {
    { vulkan_module_type, RUNTIME_MODULE_ACTIVATOR_NAME }, {}
};
#endif
