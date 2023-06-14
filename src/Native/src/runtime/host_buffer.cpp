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
#include "../kernels/stackvm/optimized/opt_ops.h"
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>

using namespace nncase;
using namespace nncase::runtime;

mapped_buffer::mapped_buffer() noexcept : buffer_(nullptr), span_() {}

mapped_buffer::mapped_buffer(host_buffer_t buffer,
                             gsl::span<gsl::byte> span) noexcept
    : buffer_(std::move(buffer)), span_(span) {}

mapped_buffer::mapped_buffer(mapped_buffer &&other) noexcept
    : buffer_(std::move(other.buffer_)), span_(other.span_) {}

mapped_buffer::~mapped_buffer() { unmap().expect("unmap failed"); }

mapped_buffer &mapped_buffer::operator=(mapped_buffer &&other) noexcept {
    unmap().expect("unmap failed");
    buffer_ = std::move(other.buffer_);
    span_ = other.span_;
    return *this;
}

result<void> mapped_buffer::unmap() noexcept {
    if (!buffer_.empty())
        try_(buffer_->unmap());
    buffer_ = nullptr;
    return ok();
}

void mapped_buffer::release() noexcept { buffer_ = nullptr; }

host_buffer_node::host_buffer_node(size_t size_bytes,
                                   buffer_allocator &allocator,
                                   host_sync_status_t host_sync_status)
    : buffer_node(size_bytes, allocator), host_sync_status_(host_sync_status) {}

result<mapped_buffer> host_buffer_node::map(map_access_t access) noexcept {
    if (host_sync_status_ == host_sync_status_t::need_invalidate) {
        try_(sync(sync_invalidate));
    }
    try_var(span, map_core(access));
    access_history_.push(access);
    return ok(mapped_buffer(this, span));
}

result<void> host_buffer_node::unmap() noexcept {
    auto last_access = access_history_.top();
    try_(unmap_core(last_access));
    if (last_access & map_write) {
        auto status = host_sync_status_;
        CHECK_WITH_ERR(status == host_sync_status_t::valid ||
                           status == host_sync_status_t::need_write_back,
                       std::errc::operation_not_permitted);
        host_sync_status_ = host_sync_status_t::need_write_back;
    }
    access_history_.pop();
    return ok();
}

result<void> host_buffer_node::sync(sync_op_t op, bool force) noexcept {
    if (!has_physical_address())
        return ok();

    bool do_sync = false;
    if (force) {
        do_sync = true;
    } else {
        auto status = host_sync_status_;
        if (op == sync_write_back) {
            CHECK_WITH_ERR(status == host_sync_status_t::valid ||
                               status == host_sync_status_t::need_write_back,
                           std::errc::operation_not_permitted);
            if (status == host_sync_status_t::need_write_back)
                do_sync = true;
        } else {
            CHECK_WITH_ERR(status == host_sync_status_t::valid ||
                               status == host_sync_status_t::need_invalidate,
                           std::errc::operation_not_permitted);
            if (status == host_sync_status_t::need_invalidate)
                do_sync = true;
        }
    }

    if (do_sync) {
        try_(sync_core(op));
        host_sync_status_ = host_sync_status_t::valid;
    }

    return ok();
}

result<void>
host_buffer_node::copy_to(buffer_t dest, size_t src_start, size_t dest_start,
                          datatype_t datatype, gsl::span<const size_t> shape,
                          gsl::span<const size_t> src_strides,
                          gsl::span<const size_t> dest_strides) noexcept {
    axes_t begins(shape.size(), 0);
    axes_t ends(shape.size(), 0);
    axes_t strides(shape.size(), 1);
    for (size_t i = 0; i < ends.size(); i++) {
        ends[i] = (int64_t)shape[i];
    }

    try_var(dest_host, dest.as<host_buffer_t>());
    try_var(src_map, map(map_read));
    try_var(dest_map, dest_host->map(map_write));
    return kernels::stackvm::optimized::slice(
        datatype, src_map.buffer().data() + src_start * datatype->size_bytes(),
        dest_map.buffer().data() + dest_start * datatype->size_bytes(), shape,
        src_strides, dest_strides, begins, ends, strides,
        kernels::default_kernel_context());
}

result<mapped_buffer> host_buffer_slice::map(map_access_t access) noexcept {
    checked_try_var(src_buffer, buffer()->map(access));
    mapped_buffer slice(buffer(),
                        src_buffer.buffer().subspan(start(), size_bytes()));
    src_buffer.release();
    return ok(std::move(slice));
}

result<void> host_buffer_slice::unmap() noexcept { return buffer()->unmap(); }

result<void> host_buffer_slice::sync(sync_op_t op, bool force) noexcept {
    return buffer()->sync(op, force);
}

bool host_buffer_slice::has_physical_address() const noexcept {
    return buffer()->has_physical_address();
}

result<uintptr_t> host_buffer_slice::physical_address() noexcept {
    checked_try_var(addr, buffer()->physical_address());
    return ok(addr + start());
}
