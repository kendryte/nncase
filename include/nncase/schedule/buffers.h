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
#pragma once
#include <nncase/ir/connectors.h>
#include <nncase/ir/ir_types.h>
#include <nncase/runtime/datatypes.h>
#include <optional>
#include <vector>

namespace nncase::schedule
{
class logical_buffer;
class physical_buffer;

struct memory_span
{
    size_t start;
    size_t size;

    size_t end() const noexcept { return start + size; }
};

struct sub_buffer_desc
{
    logical_buffer *parent = nullptr;
    size_t offset;
    ir::shape_t shape;
};

struct buffer_lifetime
{
    size_t used_count;
    size_t birth;
    size_t age;

    bool is_alive() const noexcept { return used_count > 0; }
    size_t end() const noexcept { return birth + age; }
};

class NNCASE_API logical_buffer
{
public:
    logical_buffer(size_t id, ir::output_connector &owner, memory_location_t location)
        : id_(id), owner_(owner), memory_location_(location), physical_(nullptr) { }

    size_t id() const noexcept { return id_; }
    ir::output_connector &owner() const noexcept { return owner_; }
    datatype_t type() const noexcept { return owner_.type(); }
    const ir::shape_t &shape() const noexcept { return owner_.shape(); }

    const std::optional<sub_buffer_desc> &parent() const noexcept { return parent_; }
    std::optional<sub_buffer_desc> &parent() noexcept { return parent_; }

    const ir::shape_t &strides_shape() const noexcept { return strides_shape_; }
    ir::shape_t &strides_shape() noexcept { return strides_shape_; }

    const buffer_lifetime &lifetime() const noexcept { return lifetime_; }
    buffer_lifetime &lifetime() noexcept { return lifetime_; }

    physical_buffer *physical() const noexcept { return physical_; }
    physical_buffer *&physical() noexcept { return physical_; }

    memory_location_t memory_location() const noexcept { return memory_location_; }
    memory_location_t &memory_location() noexcept { return memory_location_; }

    bool no_action_concat_with_strides() const noexcept { return no_action_concat_with_strides_; }
    bool &no_action_concat_with_strides() noexcept { return no_action_concat_with_strides_; }

private:
    size_t id_;
    ir::output_connector &owner_;
    memory_location_t memory_location_;
    std::optional<sub_buffer_desc> parent_;
    ir::shape_t strides_shape_;
    buffer_lifetime lifetime_ {};
    physical_buffer *physical_;
    bool no_action_concat_with_strides_ = false;
};

class NNCASE_API physical_buffer
{
public:
    physical_buffer(size_t id, logical_buffer &owner)
        : id_(id), owner_(owner) { }

    size_t id() const noexcept { return id_; }
    logical_buffer &owner() const noexcept { return owner_; }

    const buffer_lifetime &lifetime() const noexcept { return owner().lifetime(); }
    buffer_lifetime &lifetime() noexcept { return owner().lifetime(); }

    std::span<logical_buffer *const> logical_buffers() const noexcept { return logical_buffers_; }
    std::vector<logical_buffer *> &logical_buffers() noexcept { return logical_buffers_; }

    memory_span allocation() const noexcept { return allocation_; }
    memory_span &allocation() noexcept { return allocation_; }

    size_t alignment() const noexcept { return alignment_; }
    void alignment(size_t value) { alignment_ = value; }

private:
    size_t id_;
    logical_buffer &owner_;
    std::vector<logical_buffer *> logical_buffers_;
    memory_span allocation_;
    size_t alignment_ = 8;
};
}
