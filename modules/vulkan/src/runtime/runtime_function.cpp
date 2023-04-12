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
#include "runtime_function.h"
#include "vulkan_error.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

vulkan_runtime_function::~vulkan_runtime_function() { free_vulkan_resources(); }

vulkan_runtime_module &vulkan_runtime_function::module() const noexcept {
    return static_cast<vulkan_runtime_module &>(runtime_function::module());
}

result<void> vulkan_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    input_pool_size_ = context.header().input_pool_size;
    output_pool_size_ = context.header().output_pool_size;
    text_ = context.module_init_context().section(".text").subspan(
        context.header().entrypoint, context.header().text_size);

    try_(initialize_vulkan_device());
    try_(initialize_vulkan_memory());
    try_(initialize_vulkan_commands());
    return ok();
}

result<runtime_tensor>
vulkan_runtime_function::allocate_input_tensor(size_t index) noexcept {
    return host_runtime_tensor::create(input_desc(index).datatype,
                                       input_shape(index));
}

result<runtime_tensor>
vulkan_runtime_function::allocate_output_tensor(size_t index) noexcept {
    return host_runtime_tensor::create(output_desc(index).datatype,
                                       output_shape(index));
}

result<void>
vulkan_runtime_function::validate_input_tensor(NNCASE_UNUSED size_t index,
                                               runtime_tensor tensor) noexcept {
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> vulkan_runtime_function::validate_output_tensor(
    NNCASE_UNUSED size_t index, runtime_tensor tensor) noexcept {
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> vulkan_runtime_function::initialize_vulkan_device() noexcept {
    return ok();
}

result<void> vulkan_runtime_function::initialize_vulkan_memory() noexcept {
    if (input_pool_size_) {
        try_set(input_buffer_,
                module().allocate_vulkan_buffer(input_pool_size_));
        try_set(input_mem_, module().allocate_vulkan_memory(
                                {vk::MemoryPropertyFlagBits::eHostVisible,
                                 vk::MemoryPropertyFlagBits::eHostCached,
                                 {}},
                                input_buffer_));
        try_(module().bind_vulkan_buffer(input_buffer_, input_mem_));
    }

    if (output_pool_size_) {
        try_set(output_buffer_,
                module().allocate_vulkan_buffer(output_pool_size_));
        try_set(output_mem_, module().allocate_vulkan_memory(
                                 {vk::MemoryPropertyFlagBits::eHostVisible,
                                  vk::MemoryPropertyFlagBits::eHostCached,
                                  {}},
                                 output_buffer_));
        try_(module().bind_vulkan_buffer(output_buffer_, output_mem_));
    }

    return ok();
}

result<void> vulkan_runtime_function::initialize_vulkan_commands() noexcept {
    vk::CommandBufferAllocateInfo cmdb_cinfo(
        module().command_pool(), vk::CommandBufferLevel::ePrimary, 1);
    try_var(cmdbs, vk::to_result(
                       module().device().allocateCommandBuffers(cmdb_cinfo)));
    cmd_buffer_ = cmdbs[0];

    vk::CommandBufferBeginInfo cmdb_info;
    try_(vk::to_result(cmd_buffer_.begin(cmdb_info)));
    try_(visit(text_));
    try_(vk::to_result(cmd_buffer_.end()));
    return ok();
}

result<void> vulkan_runtime_function::preprocess_inputs() noexcept {
    try_var(dest, vk::to_result(module().device().mapMemory(
                      input_mem_, 0, VK_WHOLE_SIZE, {})));

    for (size_t i = 0; i < inputs_size(); i++) {
        try_var(src_tensor, device_input_tensor(i));
        try_var(src_map, hrt::map(src_tensor, hrt::map_read));
        auto &desc = input_desc(i);
        memcpy((uint8_t *)dest + desc.start, src_map.buffer().data(),
               desc.size);
    }

    vk::MappedMemoryRange range(input_mem_, 0, VK_WHOLE_SIZE);
    try_(vk::to_result(module().device().flushMappedMemoryRanges(range)));
    module().device().unmapMemory(input_mem_);
    return ok();
}

result<void> vulkan_runtime_function::invoke_core() noexcept {
    try_(preprocess_inputs());

    vk::SubmitInfo si({}, {}, cmd_buffer_, {});
    try_(vk::to_result(module().compute_queue().submit(si)));
    try_(vk::to_result(module().compute_queue().waitIdle()));
    try_(vk::to_result(module().device().waitIdle()));

    assert(buffer_refs_.empty());
    assert(buffer_copies_.empty());
    assert(buffer_barriers_.empty());

    try_(postprocess_outputs());
    return ok();
}

result<void> vulkan_runtime_function::postprocess_outputs() noexcept {
    try_var(src, vk::to_result(module().device().mapMemory(output_mem_, 0,
                                                           VK_WHOLE_SIZE, {})));
    vk::MappedMemoryRange range(output_mem_, 0, VK_WHOLE_SIZE);
    try_(vk::to_result(module().device().invalidateMappedMemoryRanges(range)));

    for (size_t i = 0; i < outputs_size(); i++) {
        try_var(dest_tensor, device_output_tensor(i));
        try_var(dest_map, hrt::map(dest_tensor, hrt::map_write));
        auto &desc = output_desc(i);
        memcpy(dest_map.buffer().data(), (const uint8_t *)src + desc.start,
               desc.size);
    }

    module().device().unmapMemory(output_mem_);
    return ok();
}

result<vulkan_runtime_function::buffer_ref>
vulkan_runtime_function::pop_buffer_ref() noexcept {
    if (buffer_refs_.empty())
        return err(std::errc::result_out_of_range);
    auto buffer_ref = std::move(buffer_refs_.back());
    buffer_refs_.pop_back();
    return ok(std::move(buffer_ref));
}

void vulkan_runtime_function::free_vulkan_resources() noexcept {
    if (auto device = module().device()) {
        if (module().command_pool())
            device.freeCommandBuffers(module().command_pool(), cmd_buffer_);
        device.destroyBuffer(input_buffer_);
        device.destroyBuffer(output_buffer_);
        device.freeMemory(input_mem_);
        device.freeMemory(output_mem_);
    }
}
