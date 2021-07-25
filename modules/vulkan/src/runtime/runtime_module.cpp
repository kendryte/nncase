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
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> vulkan_runtime_module::initialize_core(runtime_module_init_context &context) noexcept
{
    assert(context.is_section_pinned());
    auto descs = context.section(DESCRIPTORS_SECTION_NAME).as_span<const uint32_t>();
    descriptor_sets_ = descs[0];
    descriptors_ = descs[1];
    rdata_ = context.section(".rdata");
    text_ = context.section(".text");
    shader_ = context.section(".shader");

    try_(initialize_vulkan());
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
    try_(initialize_vulkan_instance());
    try_(initialize_vulkan_device());
    try_(initialize_vulkan_memory());
    try_(initialize_vulkan_commands());
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_commands() noexcept
{
    vk::CommandBufferBeginInfo cmdb_info;
    try_(vk::to_result(cmd_buffer_.begin(cmdb_info)));
    try_(visit(text_));
    try_(vk::to_result(cmd_buffer_.end()));
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_instance() noexcept
{
    vk::ApplicationInfo app_info("nncase.runtime", 1, "nncase", 1, VK_API_VERSION_1_1);
    vk::InstanceCreateInfo create_info({}, &app_info);
    try_set(instance_, vk::to_result(vk::createInstance(create_info)));
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_device() noexcept
{
    try_set(physical_device_, select_physical_device());
    auto queue_families = physical_device_.getQueueFamilyProperties();
    try_set(compute_queue_index_, select_queue_family(queue_families, { vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, vk::QueueFlagBits::eTransfer }));

    float priorities[] = { 0.0f };
    vk::DeviceQueueCreateInfo queue_create_info({}, compute_queue_index_, 1, priorities);
    vk::DeviceCreateInfo device_create_info({}, queue_create_info);
    try_set(device_, vk::to_result(physical_device_.createDevice(device_create_info)));
    compute_queue_ = device_.getQueue(compute_queue_index_, 0);

    vk::DescriptorPoolSize descp_size(vk::DescriptorType::eStorageBuffer, descriptors_);
    vk::DescriptorPoolCreateInfo descp_cinfo({}, descriptor_sets_, descp_size);
    try_set(buffer_desc_pool_, vk::to_result(device_.createDescriptorPool(descp_cinfo)));

    vk::CommandPoolCreateInfo cmdp_cinfo({}, compute_queue_index_);
    try_var(cmdp, vk::to_result(device_.createCommandPool(cmdp_cinfo)));
    vk::CommandBufferAllocateInfo cmdb_cinfo(cmdp, vk::CommandBufferLevel::ePrimary, 1);
    try_var(cmdbs, vk::to_result(device_.allocateCommandBuffers(cmdb_cinfo)));
    cmd_buffer_ = cmdbs[0];
    return ok();
}

result<void> vulkan_runtime_module::initialize_vulkan_memory() noexcept
{
    auto input_mem = mempool(mem_input);
    if (input_mem.size)
        try_set(input_mem_, allocate_vulkan_memory({ vk::MemoryPropertyFlagBits::eHostVisible, vk::MemoryPropertyFlagBits::eHostCached, {} }, input_mem.size));

    auto output_mem = mempool(mem_output);
    if (output_mem.size)
        try_set(output_mem_, allocate_vulkan_memory({ vk::MemoryPropertyFlagBits::eHostVisible, vk::MemoryPropertyFlagBits::eHostCached, {} }, output_mem.size));

    auto data_mem = mempool(mem_data);
    if (data_mem.size)
        try_set(data_mem_, allocate_vulkan_memory({ {}, vk::MemoryPropertyFlagBits::eDeviceLocal, {} }, data_mem.size));
    return ok();
}

result<vk::DeviceMemory> vulkan_runtime_module::allocate_vulkan_memory(const select_options<vk::MemoryPropertyFlagBits> &options, size_t required_size) noexcept
{
    auto properties = physical_device_.getMemoryProperties();
    try_var(type_index, select_memory_type(properties, options, required_size));
    vk::MemoryAllocateInfo allocate(static_cast<vk::DeviceSize>(required_size), static_cast<uint32_t>(type_index));
    return vk::to_result(device_.allocateMemory(allocate));
}

result<vk::PhysicalDevice> vulkan_runtime_module::select_physical_device() noexcept
{
    vk::PhysicalDevice *intergrated = nullptr;

    try_var(devices, vk::to_result(instance_.enumeratePhysicalDevices()));
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

result<void> vulkan_runtime_module::run_core() noexcept
{
    try_(preprocess_inputs());

    vk::SubmitInfo si({}, {}, cmd_buffer_, {});
    try_(vk::to_result(compute_queue_.submit(si)));
    try_(vk::to_result(compute_queue_.waitIdle()));

    try_(postprocess_outputs());
    return ok();
}

result<void> vulkan_runtime_module::preprocess_inputs() noexcept
{
    try_var(dest, vk::to_result(device_.mapMemory(input_mem_, 0, VK_WHOLE_SIZE, {})));

    for (size_t i = 0; i < inputs_size(); i++)
    {
        try_var(src_tensor, device_input_tensor(i));
        try_var(src_map, hrt::map(src_tensor, hrt::map_read));
        auto &desc = input_desc(i);
        memcpy((uint8_t *)dest + desc.start, src_map.buffer().data(), desc.size);
    }

    vk::MappedMemoryRange range(input_mem_, 0, VK_WHOLE_SIZE);
    try_(vk::to_result(device_.flushMappedMemoryRanges(range)));
    device_.unmapMemory(input_mem_);
    return ok();
}

result<void> vulkan_runtime_module::postprocess_outputs() noexcept
{
    try_var(src, vk::to_result(device_.mapMemory(output_mem_, 0, VK_WHOLE_SIZE, {})));
    vk::MappedMemoryRange range(output_mem_, 0, VK_WHOLE_SIZE);
    try_(vk::to_result(device_.invalidateMappedMemoryRanges(range)));

    for (size_t i = 0; i < outputs_size(); i++)
    {
        try_var(dest_tensor, device_output_tensor(i));
        try_var(dest_map, hrt::map(dest_tensor, hrt::map_write));
        auto &desc = output_desc(i);
        memcpy(dest_map.buffer().data(), (const uint8_t *)src + desc.start, desc.size);
    }

    device_.unmapMemory(output_mem_);
    return ok();
}

result<vk::Buffer> vulkan_runtime_module::pop_buffer() noexcept
{
    if (buffers_.empty())
        return err(std::errc::result_out_of_range);
    auto buffer = std::move(buffers_.back());
    buffers_.pop_back();
    return ok(std::move(buffer));
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
