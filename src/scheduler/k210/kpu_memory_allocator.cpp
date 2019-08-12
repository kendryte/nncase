/* Copyright 2019 Canaan Inc.
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
#include <ir/op_utils.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <scheduler/k210/kpu_memory_allocator.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::scheduler::k210;

kpu_memory_allocator::kpu_memory_allocator()
    : memory_allocator(64, 2 * 1024 * 1024)
{
}

size_t kpu_memory_allocator::get_bytes(datatype_t type, const ir::shape_t &shape) const
{
    if (type != dt_uint8)
        throw std::invalid_argument("KPU only support uint8 data");
    return runtime::k210::get_kpu_bytes(ir::to(shape));
}
