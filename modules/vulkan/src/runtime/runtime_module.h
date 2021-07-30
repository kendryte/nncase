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
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/vulkan/op_reader.h>
#include <nncase/runtime/vulkan/runtime_module.h>
#include <vulkan/vulkan.hpp>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

class vulkan_runtime_module : public runtime_module, private op_visitor
{
    template <class T>
    struct select_options
    {
        T requried;
        T preferred;
        T not_preferred;
    };

    struct buffer_ref
    {
        vk::Buffer buffer;
        size_t start;
        size_t size;
    };

public:
    virtual ~vulkan_runtime_module();

protected:
    result<void> initialize_core(runtime_module_init_context &context) noexcept override;
    result<runtime_tensor> allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor> allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> run_core() noexcept override;

    using op_visitor::visit;
    result<void> visit(const ldbuf_op_t &op) noexcept override;
    result<void> visit(const ldbufbarrier_op_t &op) noexcept override;
    result<void> visit(const ldbufcopy_op_t &op) noexcept override;
    result<void> visit(const copybuf_op_t &op) noexcept override;
    result<void> visit(const ldpipeline_op_t &op) noexcept override;
    result<void> visit(const dispatch_op_t &op) noexcept override;
    result<void> visit(const barrier_op_t &op) noexcept override;

private:
    result<void> initialize_vulkan() noexcept;
    result<void> initialize_vulkan_instance() noexcept;
    result<void> initialize_vulkan_device() noexcept;
    result<void> initialize_vulkan_memory() noexcept;
    result<void> initialize_vulkan_commands() noexcept;

    result<vk::PhysicalDevice> select_physical_device() noexcept;
    result<uint32_t> select_queue_family(const std::vector<vk::QueueFamilyProperties> &families, const select_options<vk::QueueFlagBits> options) noexcept;
    result<size_t> select_memory_type(const vk::PhysicalDeviceMemoryProperties &properties, const select_options<vk::MemoryPropertyFlagBits> &options, size_t required_size) noexcept;
    result<vk::DeviceMemory> allocate_vulkan_memory(const select_options<vk::MemoryPropertyFlagBits> &options, vk::Buffer buffer) noexcept;
    result<vk::Buffer> allocate_vulkan_buffer(size_t required_size) noexcept;
    result<void> bind_vulkan_buffer(vk::Buffer buffer, vk::DeviceMemory memory) noexcept;

    result<buffer_ref> pop_buffer_ref() noexcept;
    result<void> preprocess_inputs() noexcept;
    result<void> postprocess_outputs() noexcept;

    void free_vulkan_resources() noexcept;

private:
    uint32_t descriptors_;
    uint32_t descriptor_sets_;
    gsl::span<const gsl::byte> rdata_;
    gsl::span<const gsl::byte> text_;
    gsl::span<const gsl::byte> shader_;
    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    uint32_t compute_queue_index_;
    vk::Queue compute_queue_;
    vk::Buffer input_buffer_;
    vk::Buffer output_buffer_;
    vk::Buffer data_buffer_;
    vk::DeviceMemory input_mem_;
    vk::DeviceMemory output_mem_;
    vk::DeviceMemory data_mem_;
    std::vector<buffer_ref> buffer_refs_;
    std::vector<vk::Pipeline> pipelines_owner_;
    vk::DescriptorPool buffer_desc_pool_;
    vk::CommandPool cmd_pool_;
    vk::CommandBuffer cmd_buffer_;
    std::vector<vk::BufferMemoryBarrier> buffer_barriers_;
    std::vector<vk::BufferCopy> buffer_copies_;
};

END_NS_NNCASE_RT_MODULE
