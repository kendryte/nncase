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
#include <nncase/runtime/vulkan/runtime_module.h>
#include <vulkan/vulkan.hpp>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

template <class T> struct select_options {
    T requried;
    T preferred;
    T not_preferred;
};

class vulkan_context {
  public:
    ~vulkan_context();

    vk::Instance instance() const noexcept { return instance_; }
    vk::PhysicalDevice physical_device() const noexcept {
        return physical_device_;
    }
    vk::Device device() const noexcept { return device_; }
    uint32_t compute_queue_index() const noexcept {
        return compute_queue_index_;
    }
    vk::Queue compute_queue() const noexcept { return compute_queue_; }

    static result<vulkan_context *> get() noexcept;

  private:
    result<void> initialize_vulkan() noexcept;
    result<void> initialize_vulkan_instance() noexcept;
    result<void> initialize_vulkan_device() noexcept;

    result<vk::PhysicalDevice> select_physical_device() noexcept;
    result<uint32_t> select_queue_family(
        const std::vector<vk::QueueFamilyProperties> &families,
        const select_options<vk::QueueFlagBits> options) noexcept;
    result<size_t> select_memory_type(
        const vk::PhysicalDeviceMemoryProperties &properties,
        const select_options<vk::MemoryPropertyFlagBits> &options,
        size_t required_size) noexcept;

    void free_vulkan_resources() noexcept;

  private:
    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    uint32_t compute_queue_index_;
    vk::Queue compute_queue_;
};

END_NS_NNCASE_RT_MODULE
