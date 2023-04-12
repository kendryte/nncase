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

result<void> vulkan_runtime_function::visit(const copybuf_op_t &op) noexcept {
    try_var(output, pop_buffer_ref());
    try_var(input, pop_buffer_ref());

    auto unused_regions = buffer_copies_.size() - op.regions;
    auto regions = buffer_copies_.data() + unused_regions;
    for (size_t i = 0; i < op.regions; i++) {
        regions->srcOffset += input.start;
        regions->dstOffset += output.start;
    }

    cmd_buffer_.copyBuffer(input.buffer, output.buffer, {op.regions, regions});
    buffer_copies_.resize(unused_regions);
    return ok();
}
