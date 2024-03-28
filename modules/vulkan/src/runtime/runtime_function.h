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
#include "runtime_module.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/runtime_function.h>
#include <nncase/runtime/vulkan/op_reader.h>
#include <vulkan/vulkan.hpp>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

class vulkan_runtime_function : public runtime_function, private op_visitor {
    struct buffer_ref {
        vk::Buffer buffer;
        size_t start;
        size_t size;
    };

  public:
    using runtime_function::runtime_function;
    virtual ~vulkan_runtime_function();

    vulkan_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<runtime_tensor>
    allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor>
    allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index,
                                       runtime_tensor tensor) noexcept override;
    result<void>
    validate_output_tensor(size_t index,
                           runtime_tensor tensor) noexcept override;
    result<void> invoke_core() noexcept override;

    using op_visitor::visit;
    result<void> visit(const ldbuf_op_t &op) noexcept override;
    result<void> visit(const ldbufbarrier_op_t &op) noexcept override;
    result<void> visit(const ldbufcopy_op_t &op) noexcept override;
    result<void> visit(const copybuf_op_t &op) noexcept override;
    result<void> visit(const ldpipeline_op_t &op) noexcept override;
    result<void> visit(const dispatch_op_t &op) noexcept override;
    result<void> visit(const barrier_op_t &op) noexcept override;

  private:
    result<void> initialize_vulkan_device() noexcept;
    result<void> initialize_vulkan_memory() noexcept;
    result<void> initialize_vulkan_commands() noexcept;

    result<buffer_ref> pop_buffer_ref() noexcept;
    result<void> preprocess_inputs() noexcept;
    result<void> postprocess_outputs() noexcept;

    void free_vulkan_resources() noexcept;

  private:
    uint32_t input_pool_size_;
    uint32_t output_pool_size_;
    std::span<const std::byte> text_;
    vk::Buffer input_buffer_;
    vk::Buffer output_buffer_;
    vk::DeviceMemory input_mem_;
    vk::DeviceMemory output_mem_;
    std::vector<buffer_ref> buffer_refs_;
    vk::CommandBuffer cmd_buffer_;
    std::vector<vk::BufferMemoryBarrier> buffer_barriers_;
    std::vector<vk::BufferCopy> buffer_copies_;
};

END_NS_NNCASE_RT_MODULE
