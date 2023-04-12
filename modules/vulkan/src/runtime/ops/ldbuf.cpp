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

result<void> vulkan_runtime_function::visit(const ldbuf_op_t &op) noexcept {
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

    buffer_refs_.emplace_back(
        buffer_ref{dev_buf, op.memory.start, op.memory.size});
    return ok();
}
