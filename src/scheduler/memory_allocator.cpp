/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/op_utils.h>
#include <scheduler/freelist.h>
#include <scheduler/memory_allocator.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::scheduler;

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

memory_node::memory_node(memory_allocator &allocator, size_t birth, size_t valid_size, size_t size)
    : allocator_(allocator), birth_(birth), valid_size_(valid_size), size_(size), use_count_(0), age_(0)
{
}

void memory_node::add_ref()
{
    ++use_count_;
}

void memory_node::release()
{
    if (--use_count_ == 0)
        allocator_.free(*this);

    if (use_count_ < 0)
        throw std::runtime_error("Memory node has been freed");
}

void memory_node::grow_age()
{
    age_++;
}

memory_allocator::memory_allocator(size_t alignment, std::optional<size_t> fixed_size)
    : alignment_(alignment), age_(0), fixed_size_(fixed_size)
{
}

memory_node &memory_allocator::allocate(size_t size)
{
    auto aligned_size = align(size, alignment_);
    //auto free_node = freelist_.allocate(aligned_size);
    auto &node = nodes_.emplace_back(*this, age_, size, aligned_size);
    node.add_ref();
    return node;
}

void memory_allocator::free(memory_node &node)
{
    //freelist_.free(free_memory_node { node.start(), node.size() });
}

void memory_allocator::grow_age()
{
    age_++;
    for (auto &n : nodes_)
    {
        if (n.used())
            n.grow_age();
    }
}

void memory_allocator::finish(uint32_t max_solve_secs)
{
    bool solved = false;
    size_t max_usage = 0;

    freelist fl(std::nullopt);
    size_t age = 0;
    size_t allocated_nodes = 0;
    while (allocated_nodes < nodes_.size())
    {
        for (auto &n : nodes_)
        {
            if (age == n.birth())
            {
                auto alloc_node = fl.allocate(n.size());
                n.start(alloc_node.start);
                allocated_nodes++;
            }

            if (age == n.birth() + n.age())
            {
                fl.free({ n.start(), n.size() });
            }
        }

        age++;
    }

    max_usage = fl.max_usage();

    if (fixed_size_ && max_usage > *fixed_size_)
        throw std::runtime_error("KPU allocator cannot allocate more memory.");
    max_usage_ = max_usage;
}

size_t memory_allocator::get_bytes(datatype_t type, const hlir::shape_t &shape) const
{
    return runtime::get_bytes(type) * xt::compute_size(shape);
}
