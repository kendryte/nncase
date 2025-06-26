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
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/copy.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/device_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;

device_mapped_buffer::device_mapped_buffer() noexcept
    : buffer_(nullptr), span_() {}

device_mapped_buffer::device_mapped_buffer(device_buffer_t buffer,
                                           std::span<std::byte> span) noexcept
    : buffer_(std::move(buffer)), span_(span) {}

device_mapped_buffer::device_mapped_buffer(
    device_mapped_buffer &&other) noexcept
    : buffer_(std::move(other.buffer_)), span_(other.span_) {}

device_mapped_buffer &
device_mapped_buffer::operator=(device_mapped_buffer &&other) noexcept {
    unmap().expect("unmap failed");
    buffer_ = std::move(other.buffer_);
    span_ = other.span_;
    return *this;
}

result<void> device_mapped_buffer::unmap() noexcept {
    if (!buffer_.empty())
        try_(buffer_->unmap());
    buffer_ = nullptr;
    return ok();
}

void device_mapped_buffer::release() noexcept { buffer_ = nullptr; }

device_buffer_node::device_buffer_node(size_t size_bytes,
                                       buffer_allocator &allocator,
                                       device_sync_status_t device_sync_status)
    : buffer_node(size_bytes, allocator),
      device_sync_status_(device_sync_status) {}

result<device_mapped_buffer>
device_buffer_node::map(map_access_t access) noexcept {
    if (device_sync_status_ == device_sync_status_t::need_invalidate) {
        try_(sync(sync_invalidate));
    }
    try_var(span, map_core(access));
    access_history_.push(access);
    return ok(device_mapped_buffer(this, span));
}

result<void> device_buffer_node::unmap() noexcept { return ok(); }

result<void> device_buffer_node::sync([[maybe_unused]] sync_op_t op,
                                      [[maybe_unused]] bool force) noexcept {
    return ok();
}

result<void> device_buffer_node::copy_to(
    [[maybe_unused]] buffer_t dest, [[maybe_unused]] size_t src_start,
    [[maybe_unused]] size_t dest_start, [[maybe_unused]] datatype_t datatype,
    [[maybe_unused]] std::span<const size_t> shape,
    [[maybe_unused]] std::span<const size_t> src_strides,
    [[maybe_unused]] std::span<const size_t> dest_strides) noexcept {
    return ok();
}

result<int> device_buffer_node::device_type() noexcept {
    return ok(device_type_);
}
result<int> device_buffer_node::device_id() noexcept { return ok(device_id_); }

result<device_mapped_buffer>
device_buffer_slice::map(map_access_t access) noexcept {
    return buffer()->map(access);
}

result<void> device_buffer_slice::unmap() noexcept { return buffer()->unmap(); }

result<void> device_buffer_slice::sync(sync_op_t op, bool force) noexcept {
    return buffer()->sync(op, force);
}

bool device_buffer_slice::has_physical_address() const noexcept {
    return false;
}

result<uintptr_t> device_buffer_slice::physical_address() noexcept {
    return err(std::errc::not_supported);
}
