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
#include "../runtime_module.h"
#include "../vulkan_error.h"
#include <nncase/runtime/error.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> vulkan_runtime_module::visit(const ldbuf_op_t &op) noexcept
{
    vk::DeviceMemory *dev_mem;
    switch (op.memory.memory_location)
    {
    case mem_input:
        dev_mem = &input_mem_;
        break;
    case mem_output:
        dev_mem = &output_mem_;
        break;
    case mem_data:
        dev_mem = &data_mem_;
        break;
    default:
        return err(nncase_errc::invalid_memory_location);
    }

    vk::BufferCreateInfo info({}, (vk::DeviceSize)op.memory.size,
        vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive, 1, &compute_queue_index_);
    try_var(buffer, vk::to_result(device_.createBuffer(info)));
    try_(vk::to_result(device_.bindBufferMemory(buffer, *dev_mem, (vk::DeviceSize)op.memory.start)));
    buffers_owner_.emplace_back(buffer);
    buffers_.emplace_back(buffer);
    return ok();
}
