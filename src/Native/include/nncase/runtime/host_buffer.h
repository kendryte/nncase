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
#include "buffer.h"
#include <nncase/runtime/small_vector.hpp>
#include <stack>

BEGIN_NS_NNCASE_RUNTIME

class host_buffer_node;
using host_buffer_t = object_t<host_buffer_node>;

class NNCASE_API mapped_buffer {
  public:
    mapped_buffer() noexcept;
    mapped_buffer(host_buffer_t buffer, gsl::span<gsl::byte> span) noexcept;
    mapped_buffer(mapped_buffer &&other) noexcept;
    mapped_buffer(const mapped_buffer &) = delete;
    ~mapped_buffer();

    mapped_buffer &operator=(mapped_buffer &&) noexcept;
    mapped_buffer &operator=(const mapped_buffer &) = delete;

    result<void> unmap() noexcept;
    void release() noexcept;

    gsl::span<gsl::byte> buffer() const noexcept { return span_; }

  private:
    host_buffer_t buffer_;
    gsl::span<gsl::byte> span_;
};

class NNCASE_API host_buffer_node : public buffer_node {
    DEFINE_OBJECT_KIND(buffer_node, object_host_buffer);

  public:
    host_buffer_node(
        size_t size_bytes, buffer_allocator &allocator,
        host_sync_status_t host_sync_status = host_sync_status_t::valid);

    host_sync_status_t host_sync_status() const noexcept {
        return host_sync_status_;
    }

    void host_sync_status(host_sync_status_t status) noexcept {
        host_sync_status_ = status;
    }

    result<mapped_buffer> map(map_access_t access) noexcept;
    result<void> unmap() noexcept;
    result<void> sync(sync_op_t op, bool force = false) noexcept;

    virtual bool has_physical_address() const noexcept = 0;
    virtual result<uintptr_t> physical_address() noexcept = 0;

    result<void>
    copy_to(buffer_t dest, size_t src_start, size_t dest_start,
            datatype_t datatype, gsl::span<const size_t> shape,
            gsl::span<const size_t> src_strides,
            gsl::span<const size_t> dest_strides) noexcept override;

  protected:
    virtual result<gsl::span<gsl::byte>> map_core(map_access_t access) = 0;
    virtual result<void> unmap_core(map_access_t access) = 0;
    virtual result<void> sync_core(sync_op_t op) = 0;

  private:
    host_sync_status_t host_sync_status_;
    std::stack<map_access_t, itlib::small_vector<map_access_t, 2>>
        access_history_;
};

class NNCASE_API host_buffer_slice : public buffer_slice {
  public:
    host_buffer_slice() noexcept = default;
    host_buffer_slice(host_buffer_t buffer) noexcept
        : buffer_slice(std::move(buffer)) {}

    host_buffer_slice(host_buffer_t buffer, size_t start,
                      size_t length) noexcept
        : buffer_slice(std::move(buffer), start, length) {}

    const host_buffer_t &buffer() const noexcept {
        return reinterpret_cast<const host_buffer_t &>(buffer_slice::buffer());
    }

    result<mapped_buffer> map(map_access_t access) noexcept;
    result<void> unmap() noexcept;
    result<void> sync(sync_op_t op, bool force = false) noexcept;

    bool has_physical_address() const noexcept;
    result<uintptr_t> physical_address() noexcept;
};

END_NS_NNCASE_RUNTIME
