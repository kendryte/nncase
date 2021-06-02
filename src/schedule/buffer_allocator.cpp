/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/op_utils.h>
#include <nncase/schedule/buffer_allocator.h>
#include <nncase/schedule/freelist.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::schedule;

namespace
{
constexpr size_t align(size_t size, size_t alignment = 8)
{
    size_t remainder = size % alignment;
    if (remainder != 0)
        return size - remainder + alignment;
    return size;
}
}

size_t buffer_allocator::get_size_in_bytes(const physical_buffer &buffer)
{
    return ir::get_bytes(buffer.owner().type(), buffer.owner().shape());
}

size_t buffer_allocator::alignment() const noexcept
{
    return 8;
}

buffer_allocator::allocated_buffer buffer_allocator::make_alloc(const physical_buffer &buffer)
{
    allocated_buffer alloc;
    alloc.buffer = &buffer;
    alloc.valid_size = get_size_in_bytes(buffer);
    alloc.size = align(alloc.valid_size, alignment());
    return alloc;
}

void linear_buffer_allocator::base_offset(size_t value)
{
    max_usage_ = value;
}

void linear_buffer_allocator::mark(const physical_buffer &buffer)
{
    auto alloc = make_alloc(buffer);
    alloc.start = (uint32_t)max_usage_;
    allocations_.emplace(&buffer, alloc);
    max_usage_ += alloc.size;
}

void linear_buffer_allocator::finish()
{
}

first_fit_allocator::first_fit_allocator(std::optional<size_t> fixed_size)
    : list_(fixed_size)
{
}

void first_fit_allocator::base_offset([[maybe_unused]] size_t value)
{
    throw std::runtime_error("First fit allocator doesn't support base offset");
}

void first_fit_allocator::mark(const physical_buffer &buffer)
{
    auto age = buffer.lifetime().birth;

    // 1. Free dead buffers
    for (auto it = living_buffers_.begin(); it != living_buffers_.end();)
    {
        if ((*it)->lifetime().end() <= age)
        {
            auto &alloc = allocations_.at(*it);
            list_.free({ alloc.start, alloc.size });
            it = living_buffers_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // 2. Allocate new
    auto alloc = make_alloc(buffer);
    auto alloc_node = list_.allocate(alloc.size);
    alloc.start = alloc_node.start;
    allocations_.emplace(&buffer, alloc);
    living_buffers_.emplace_back(&buffer);
}

void first_fit_allocator::finish()
{
    max_usage_ = list_.max_usage();
}
