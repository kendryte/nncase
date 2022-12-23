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
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/schedule/k210/kpu_buffer_allocator.h>

using namespace nncase;
using namespace nncase::schedule;
using namespace nncase::schedule::k210;

kpu_buffer_allocator::kpu_buffer_allocator()
    : first_fit_allocator(2 * 1024 * 1024) {}

size_t kpu_buffer_allocator::alignment() const noexcept { return 64; }

size_t kpu_buffer_allocator::get_size_in_bytes(const logical_buffer &buffer) {
    if (buffer.type() != dt_uint8)
        throw std::invalid_argument("KPU only support uint8 datatype");
    return runtime::k210::get_kpu_bytes(buffer.shape());
}
