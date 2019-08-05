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
#include <algorithm>
#include <cassert>
#include <scheduler/freelist.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::scheduler;

freelist::freelist(std::optional<size_t> fixed_size)
    : is_fixed_(fixed_size)
{
    if (fixed_size)
    {
        free_nodes_.emplace(0, free_memory_node { 0, *fixed_size });
        heap_end_ = *fixed_size;
    }
}

void freelist::free(const free_memory_node &node)
{
    free_nodes_.emplace(node.start, node);
    auto it = free_nodes_.find(node.start);
    merge(it);
}

free_memory_node freelist::allocate(size_t size)
{
    auto free = reserve(size);

    assert(free != free_nodes_.end());
    auto node = free->second;

    if (free->second.size == size)
    {
        free_nodes_.erase(free);
    }
    else
    {
        free->second.size -= size;
        node.start = free->second.end();
        node.size = size;
    }

    return node;
}

freelist::free_nodes_t::iterator freelist::reserve(size_t size)
{
    auto free = std::find_if(free_nodes_.begin(), free_nodes_.end(), [=](const auto &node) { return node.second.size >= size; });

    // Not enough free space
    if (free == free_nodes_.end())
    {
        if (is_fixed_)
            throw std::runtime_error("Allocator has ran out of memory");

        // No free node
        if (!free_nodes_.empty())
        {
            auto last = std::prev(free_nodes_.end());
            if (last->second.end() == heap_end_)
            {
                auto enlarge = size - last->second.size;
                last->second.size += enlarge;
                heap_end_ += enlarge;
                return last;
            }
        }

        auto it = free_nodes_.emplace(heap_end_, free_memory_node { heap_end_, size });
        heap_end_ += size;
        return it.first;
    }

    return free;
}

void freelist::merge(freelist::free_nodes_t::iterator offset)
{
    auto left = std::prev(offset);
    if (left != std::end(free_nodes_) && left->second.end() == offset->second.start)
    {
        left->second.size += offset->second.size;
        free_nodes_.erase(offset);
        return merge(left);
    }

    auto right = std::next(offset);
    if (right != std::end(free_nodes_) && right->second.start == offset->second.end())
    {
        offset->second.size += right->second.size;
        free_nodes_.erase(right);
        return merge(offset);
    }
}
