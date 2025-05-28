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
#include "section.h"
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_section_context.h>

using namespace nncase;
using namespace nncase::runtime;

result<gsl::span<const gsl::byte>> runtime_section_context::get_or_read_section(
    const char *name, host_buffer_t &storage, bool allocate_shared) noexcept {
    gsl::span<const gsl::byte> src_span;
    stream_reader *sr = nullptr;
    size_t body_size;

    auto section_span_r = section(name);
    if (section_span_r.is_ok()) {
        src_span = std::move(section_span_r).unwrap();
        body_size = src_span.size_bytes();

        // Try to attach if section is pinned
        if (is_section_pinned()) {
            buffer_attach_options options{};
            options.flags = allocate_shared ? HOST_BUFFER_ATTACH_SHARED : 0;
            auto buffer_r = buffer_allocator::host().attach(
                {const_cast<gsl::byte *>(src_span.data()), src_span.size()},
                options);

            if (buffer_r.is_ok()) {
                storage = buffer_r.unwrap().as<host_buffer_t>().unwrap();
                return ok(src_span);
            } else {
                if (!allocate_shared) {
                    return buffer_r.unwrap_err();
                }
            }
        }
    } else {
        section_header header;
        try_set(sr, seek_section(name, header));
        body_size = header.body_size;
    }

    // Allocate buffer
    buffer_allocate_options options{};
    options.flags = allocate_shared ? HOST_BUFFER_ALLOCATE_SHARED
                                    : HOST_BUFFER_ALLOCATE_CPU_ONLY;
    try_var(buffer, buffer_allocator::host().allocate(body_size, options));
    storage = buffer.as<host_buffer_t>().unwrap();
    gsl::span<const gsl::byte> span;

    // Read section into buffer
    {
        try_var(mapped, storage->map(map_write));
        if (!sr) {
            memcpy(mapped.buffer().data(), src_span.data(), body_size);
        } else {
            sr->read_span(mapped.buffer());
        }

        span = allocate_shared
                   ? gsl::make_span(reinterpret_cast<const gsl::byte *>(
                                        storage->physical_address().unwrap()),
                                    body_size)
                   : mapped.buffer();
    }

    if (allocate_shared) {
        try_(storage->sync(sync_write_back));
    }

    return ok(span);
}
