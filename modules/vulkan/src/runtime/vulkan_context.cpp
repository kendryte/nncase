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
#include "vulkan_context.h"
#include "vulkan_error.h"
#include <nncase/runtime/dbg.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

vulkan_context::~vulkan_context() { free_vulkan_resources(); }

result<void> vulkan_context::initialize_vulkan() noexcept {
    checked_try(initialize_vulkan_instance());
    checked_try(initialize_vulkan_device());
    return ok();
}

const std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

result<void> checkValidationLayerSupport() {
    checked_try_var(layer_props,
                    vk::to_result(vk::enumerateInstanceLayerProperties({})));
    for (auto &layer : validation_layers) {
        bool layerFound = false;

        for (const auto &props : layer_props) {
            if (strcmp(layer, props.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return err(std::errc::no_such_device);
        }
    }

    return ok();
}

result<void> vulkan_context::initialize_vulkan_instance() noexcept {
    checked_try(checkValidationLayerSupport());

    vk::ApplicationInfo app_info("nncase.runtime", 1, "nncase", 1,
                                 VK_API_VERSION_1_1);
    vk::InstanceCreateInfo create_info({}, &app_info, validation_layers);
    checked_try_set(instance_, vk::to_result(vk::createInstance(create_info)));
    return ok();
}

result<vulkan_context *> vulkan_context::get() noexcept {
    struct vulkan_init {
        vulkan_init() : r(err(std::errc::no_such_device)) {
            r = ctx.initialize_vulkan();
        }

        vulkan_context ctx;
        result<void> r;
    };
    static vulkan_init init;
    checked_try(init.r);
    return ok(&init.ctx);
}

result<void> vulkan_context::initialize_vulkan_device() noexcept {
    checked_try_set(physical_device_, select_physical_device());
    auto queue_families = physical_device_.getQueueFamilyProperties();
    checked_try_set(
        compute_queue_index_,
        select_queue_family(queue_families, {vk::QueueFlagBits::eCompute,
                                             vk::QueueFlagBits::eGraphics,
                                             vk::QueueFlagBits::eTransfer}));

    float priorities[] = {0.0f};
    vk::DeviceQueueCreateInfo queue_create_info({}, compute_queue_index_, 1,
                                                priorities);
    vk::DeviceCreateInfo device_create_info({}, queue_create_info);
    checked_try_set(device_, vk::to_result(physical_device_.createDevice(
                                 device_create_info)));
    compute_queue_ = device_.getQueue(compute_queue_index_, 0);
    return ok();
}

result<vk::PhysicalDevice> vulkan_context::select_physical_device() noexcept {
    vk::PhysicalDevice *intergrated = nullptr;

    checked_try_var(devices,
                    vk::to_result(instance().enumeratePhysicalDevices()));
    for (auto &device : devices) {
        auto properties = device.getProperties();
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            return ok(device);
        else if (properties.deviceType ==
                 vk::PhysicalDeviceType::eIntegratedGpu)
            intergrated = &device;
    }

    if (intergrated)
        return ok(*intergrated);
    else if (!devices.empty())
        return ok(devices.front());
    else
        return err(std::errc::no_such_device);
}

result<uint32_t> vulkan_context::select_queue_family(
    const std::vector<vk::QueueFamilyProperties> &families,
    const select_options<vk::QueueFlagBits> options) noexcept {
    // 1. try required & preferred & !not_preferred
    for (uint32_t i = 0; i < families.size(); i++) {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried &&
            (flags & options.preferred) == options.preferred &&
            !(flags & options.not_preferred))
            return ok(i);
    }

    // 2. try required & preferred
    for (uint32_t i = 0; i < families.size(); i++) {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried &&
            (flags & options.preferred) == options.preferred)
            return ok(i);
    }

    // 3. try required
    for (uint32_t i = 0; i < families.size(); i++) {
        auto flags = families[i].queueFlags;
        if ((flags & options.requried) == options.requried)
            return ok(i);
    }

    std::cerr << "Cannot find available queue: " << to_string(options.requried)
              << std::endl;
    return err(std::errc::no_such_device);
}

void vulkan_context::free_vulkan_resources() noexcept {
    device_.destroy({});
    instance_.destroy({});
}
