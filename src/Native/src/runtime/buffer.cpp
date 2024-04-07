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
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_buffer.h>

using namespace nncase;
using namespace nncase::runtime;

buffer_node::buffer_node(size_t size_bytes, buffer_allocator &allocator)
    : size_bytes_(size_bytes), allocator_(allocator) {}

result<host_buffer_slice> buffer_slice::as_host() const noexcept {
    checked_try_var(host_buffer, buffer_.as<host_buffer_t>());
    return ok(host_buffer_slice(host_buffer, start_, length_));
}

result<void>
buffer_slice::copy_to(const buffer_slice &dest, datatype_t datatype,
                      std::span<const size_t> shape,
                      std::span<const size_t> src_strides,
                      std::span<const size_t> dest_strides) const noexcept {
    return buffer()->copy_to(dest.buffer(), start(), dest.start(), datatype,
                             shape, src_strides, dest_strides);
}
