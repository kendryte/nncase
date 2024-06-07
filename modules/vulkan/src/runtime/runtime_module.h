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
#include "vulkan_context.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/vulkan/op_reader.h>
#include <nncase/runtime/vulkan/runtime_module.h>
#include <vulkan/vulkan.hpp>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

class vulkan_runtime_module : public runtime_module {

  public:
    virtual ~vulkan_runtime_module();

    vk::Buffer data() const noexcept { return data_buffer_; }
    vk::Buffer rdata() const noexcept { return {}; }
    std::span<const std::byte> shader() const noexcept { return shader_; }

    vk::Device device() const noexcept { return ctx_->device(); }
    vk::CommandPool command_pool() const noexcept { return cmd_pool_; }
    uint32_t compute_queue_index() const noexcept {
        return ctx_->compute_queue_index();
    }
    vk::Queue compute_queue() const noexcept { return ctx_->compute_queue(); }
    vk::DescriptorPool buffer_desc_pool() const noexcept {
        return buffer_desc_pool_;
    }

    result<vk::DeviceMemory> allocate_vulkan_memory(
        const select_options<vk::MemoryPropertyFlagBits> &options,
        vk::Buffer buffer) noexcept;
    result<vk::Buffer> allocate_vulkan_buffer(size_t required_size) noexcept;
    result<void> bind_vulkan_buffer(vk::Buffer buffer,
                                    vk::DeviceMemory memory) noexcept;
    result<void> add_pipeline(vk::Pipeline pipeline,
                              vk::PipelineLayout pipeline_layout,
                              vk::DescriptorSetLayout set_layout) noexcept;

  protected:
    result<void> initialize_before_functions(
        runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>>
    create_function() noexcept override;

  private:
    result<void> initialize_vulkan() noexcept;
    result<void> initialize_vulkan_device() noexcept;
    result<void> initialize_vulkan_memory() noexcept;

    result<size_t> select_memory_type(
        const vk::PhysicalDeviceMemoryProperties &properties,
        const select_options<vk::MemoryPropertyFlagBits> &options,
        size_t required_size) noexcept;

    void free_vulkan_resources() noexcept;

  private:
    uint32_t descriptors_;
    uint32_t descriptor_sets_;
    std::span<const std::byte> text_;
    std::span<const std::byte> shader_;
    vulkan_context *ctx_;
    vk::Buffer data_buffer_;
    vk::DeviceMemory data_mem_;
    std::vector<vk::Pipeline> pipelines_owner_;
    std::vector<vk::PipelineLayout> pipeline_layouts_owner_;
    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts_owner_;
    vk::DescriptorPool buffer_desc_pool_;
    vk::CommandPool cmd_pool_;
};

END_NS_NNCASE_RT_MODULE
