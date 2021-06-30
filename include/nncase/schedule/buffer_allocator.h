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
#pragma once
#include "buffers.h"
#include "freelist.h"
#include <list>
#include <nncase/ir/ir_types.h>
#include <nncase/runtime/datatypes.h>
#include <optional>
#include <unordered_map>

namespace nncase::schedule
{
class NNCASE_API buffer_allocator
{
public:
    struct allocated_buffer : memory_span
    {
        const physical_buffer *buffer;
        size_t valid_size;
    };

    virtual ~buffer_allocator() = default;
    virtual void base_offset(size_t value) = 0;
    virtual void mark(const physical_buffer &buffer) = 0;
    virtual void finish() = 0;
    size_t max_usage() const noexcept { return max_usage_; }
    const std::unordered_map<const physical_buffer *, allocated_buffer> &allocations() const noexcept { return allocations_; }

    virtual size_t get_size_in_bytes(const logical_buffer &buffer);

protected:
    virtual allocated_buffer make_alloc(const physical_buffer &buffer);
    virtual size_t alignment() const noexcept;

protected:
    size_t max_usage_;
    std::unordered_map<const physical_buffer *, allocated_buffer> allocations_;
};

class NNCASE_API linear_buffer_allocator : public buffer_allocator
{
public:
    void base_offset(size_t value) override;
    void mark(const physical_buffer &buffer) override;
    void finish() override;
};

class NNCASE_API first_fit_allocator : public buffer_allocator
{
public:
    first_fit_allocator(std::optional<size_t> fixed_size = std::nullopt);

    void base_offset(size_t value) override;
    void mark(const physical_buffer &buffer) override;
    void finish() override;

private:
    freelist list_;
    std::vector<const physical_buffer *> living_buffers_;
};

using allocator_map_t = std::unordered_map<memory_location_t, buffer_allocator *>;
}
