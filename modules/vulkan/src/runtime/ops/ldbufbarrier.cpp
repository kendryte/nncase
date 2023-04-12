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
#include "../runtime_function.h"
#include "../vulkan_error.h"
#include <nncase/runtime/error.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void>
vulkan_runtime_function::visit(const ldbufbarrier_op_t &op) noexcept {
    vk::Buffer dev_buf;
    switch (op.memory.memory_location) {
    case mem_input:
        dev_buf = input_buffer_;
        break;
    case mem_output:
        dev_buf = output_buffer_;
        break;
    case mem_data:
        dev_buf = module().data();
        break;
    default:
        return err(nncase_errc::invalid_memory_location);
    }

    buffer_barriers_.emplace_back(vk::BufferMemoryBarrier(
        (vk::AccessFlagBits)op.src_access_mask,
        (vk::AccessFlagBits)op.dest_access_mask, module().compute_queue_index(),
        module().compute_queue_index(), dev_buf,
        (vk::DeviceSize)op.memory.start, (vk::DeviceSize)op.memory.size));
    return ok();
}
