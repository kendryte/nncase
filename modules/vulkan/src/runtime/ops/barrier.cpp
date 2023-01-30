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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;

result<void> vulkan_runtime_function::visit(const barrier_op_t &op) noexcept {
    CHECK_WITH_ERR(op.memory_barriers == 0, std::errc::not_supported);

    auto unused_buf_barriers = buffer_barriers_.size() - op.buffer_barriers;
    auto buf_barriers = buffer_barriers_.data() + unused_buf_barriers;

    cmd_buffer_.pipelineBarrier((vk::PipelineStageFlagBits)op.src_stage,
                                (vk::PipelineStageFlagBits)op.dest_stage,
                                (vk::DependencyFlagBits)op.dependency_flags, {},
                                {op.buffer_barriers, buf_barriers}, {});
    buffer_barriers_.resize(unused_buf_barriers);
    return ok();
}
