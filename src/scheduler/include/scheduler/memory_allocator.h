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
#pragma once
#include "freelist.h"
#include <datatypes.h>
#include <ir/ir_types.h>
#include <list>
#include <optional>

namespace nncase
{
namespace scheduler
{
    class memory_allocator;

    struct memory_allocation
    {
        memory_type_t type;
        size_t start;
        size_t size;

        size_t end() const noexcept { return start + size; }

        bool overlap(const memory_allocation &rhs) const noexcept
        {
            return type == rhs.type && (start < rhs.end() && end() > rhs.start);
        }
    };

    class memory_node
    {
    public:
        memory_node(memory_allocator &allocator, size_t start, size_t size);

        memory_node(memory_node &) = delete;
        memory_node(memory_node &&) = default;
        memory_node &operator=(memory_node &) = delete;

        long use_count() const noexcept { return use_count_; }
        size_t start() const noexcept { return start_; }
        size_t size() const noexcept { return size_; }
        size_t end() const noexcept { return start() + size(); }
        bool used() const noexcept { return use_count_; }
        size_t safe_start() const;

        void add_ref();
        void release();

    private:
        memory_allocator &allocator_;
        size_t start_;
        size_t size_;
        long use_count_;
    };

    class memory_allocator
    {
    public:
        memory_allocator(size_t alignment = 8, std::optional<size_t> fixed_size = std::nullopt);

        memory_node &allocate(size_t size);
        void free(memory_node &node);
        size_t max_usage() const noexcept { return freelist_.max_usage(); }

        virtual size_t get_bytes(datatype_t type, const ir::shape_t &shape) const;

    private:
        size_t alignment_;
        freelist freelist_;
        std::list<memory_node> nodes_;
    };
}
}
