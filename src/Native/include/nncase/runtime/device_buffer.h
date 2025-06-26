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

class device_buffer_node;
using device_buffer_t = object_t<device_buffer_node>;

class NNCASE_API device_mapped_buffer {
  public:
    device_mapped_buffer() noexcept;
    device_mapped_buffer(device_buffer_t buffer,
                         std::span<std::byte> span) noexcept;
    device_mapped_buffer(device_mapped_buffer &&other) noexcept;
    device_mapped_buffer(const device_mapped_buffer &) = delete;
    ~device_mapped_buffer();

    device_mapped_buffer &operator=(device_mapped_buffer &&) noexcept;
    device_mapped_buffer &operator=(const device_mapped_buffer &) = delete;

    virtual result<void> unmap() noexcept;
    virtual void release() noexcept;

    virtual std::span<std::byte> buffer() const noexcept { return span_; }

  private:
    device_buffer_t buffer_;
    std::span<std::byte> span_;
};

class NNCASE_API device_buffer_node : public buffer_node {
    DEFINE_OBJECT_KIND(buffer_node, object_device_buffer);

  public:
    device_buffer_node(
        size_t size_bytes, buffer_allocator &allocator,
        device_sync_status_t host_sync_status = device_sync_status_t::valid);

    virtual device_sync_status_t device_sync_status() const noexcept {
        return device_sync_status_;
    }

    virtual void device_sync_status(device_sync_status_t status) noexcept {
        device_sync_status_ = status;
    }

    virtual result<device_mapped_buffer> map(map_access_t access) noexcept;
    virtual result<void> unmap() noexcept;
    virtual result<void> sync(sync_op_t op, bool force = false) noexcept;

    virtual result<void>
    copy_to(buffer_t dest, size_t src_start, size_t dest_start,
            datatype_t datatype, std::span<const size_t> shape,
            std::span<const size_t> src_strides,
            std::span<const size_t> dest_strides) noexcept override;

    virtual result<int> device_type() noexcept;
    virtual result<int> device_id() noexcept;

  protected:
    virtual result<std::span<std::byte>> map_core(map_access_t access) = 0;
    virtual result<void> unmap_core(map_access_t access) = 0;
    virtual result<void> sync_core(sync_op_t op) = 0;

  private:
    int device_type_;
    int device_id_;
    device_sync_status_t device_sync_status_;
    std::stack<map_access_t, itlib::small_vector<map_access_t, 2>>
        access_history_;
};

class NNCASE_API device_buffer_slice : public buffer_slice {
  public:
    device_buffer_slice() noexcept = default;
    device_buffer_slice(device_buffer_t buffer) noexcept
        : buffer_slice(std::move(buffer)) {}

    device_buffer_slice(device_buffer_t buffer, size_t start,
                        size_t length) noexcept
        : buffer_slice(std::move(buffer), start, length) {}

    virtual const device_buffer_t &buffer() const noexcept {
        return reinterpret_cast<const device_buffer_t &>(
            buffer_slice::buffer());
    }

    virtual result<device_mapped_buffer> map(map_access_t access) noexcept;
    virtual result<void> unmap() noexcept;
    virtual result<void> sync(sync_op_t op, bool force = false) noexcept;

    virtual bool has_physical_address() const noexcept;
    virtual result<uintptr_t> physical_address() noexcept;
};

END_NS_NNCASE_RUNTIME
