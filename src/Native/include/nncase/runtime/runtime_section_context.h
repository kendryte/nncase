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
#include "device_buffer.h"
#include "host_buffer.h"
#include "model.h"
#include "result.h"
#include "span_reader.h"
#include "stream_reader.h"
#include <nncase/runtime/allocator.h>
#include <nncase/type.h>
#include <nncase/value.h>
using namespace nncase;
using namespace nncase::runtime;
BEGIN_NS_NNCASE_RUNTIME

struct NNCASE_API runtime_section_context {
    virtual bool is_section_pinned() const noexcept = 0;
    virtual result<std::span<const std::byte>>
    section(const char *name) noexcept = 0;
    virtual result<stream_reader *>
    seek_section(const char *name, section_header &header) noexcept = 0;

    template <class TCallable = std::nullptr_t>
    result<std::span<const std::byte>>
    get_or_read_section(const char *name, buffer_t &buffer_storage,
                        bool allocate_shared,
                        TCallable &&callable = nullptr) noexcept;

    result<std::span<const std::byte>>
    get_or_read_section(const char *name, buffer_t &buffer_storage,
                        bool allocate_shared) noexcept;

    template <class TCallable>
    result<void> read_section(const char *name, TCallable &&callable) noexcept {
        auto section_span_r = section(name);
        if (section_span_r.is_ok()) {
            span_reader sr(std::move(section_span_r).unwrap());
            return callable(sr, sr.avail());
        } else {
            section_header header;
            try_var(sr, seek_section(name, header));
            return callable(*sr, (size_t)header.body_size);
        }
    }
};

template <class TCallable>
result<std::span<const std::byte>> runtime_section_context::get_or_read_section(
    const char *name, buffer_t &buffer_storage, bool allocate_shared,
    TCallable &&callable) noexcept {
    std::span<const std::byte> src_span;
    stream_reader *sr = nullptr;
    size_t body_size;
    uint32_t alignment;
    host_buffer_t host_storage;
    device_buffer_t device_storage;

    // device buffer only worked when use stream to load kmodel.
    auto section_span_r = section(name);
    if (section_span_r.is_ok()) {
        // host_storage = buffer_storage.as<host_buffer_t>().unwrap();
        src_span = std::move(section_span_r).unwrap();
        body_size = src_span.size_bytes();
        auto header = reinterpret_cast<const section_header *>(src_span.data());
        alignment = header->alignment;

        // Try to attach if section is pinned
        if (is_section_pinned()) {
            buffer_attach_options options{};
            options.flags = allocate_shared ? HOST_BUFFER_ATTACH_SHARED : 0;
            options.alignment = alignment;
            auto buffer_r = buffer_allocator::host().attach(
                {const_cast<std::byte *>(src_span.data()), src_span.size()},
                options);

            if (buffer_r.is_ok()) {
                buffer_storage = buffer_r.unwrap().as<host_buffer_t>().unwrap();
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
        alignment = header.alignment;
        std::cout<<"\t body_size: "<<body_size<<std::endl;
    }
    std::cout << "before callable" << std::endl;
    std::span<const std::byte> span;
    if constexpr (std::is_same_v<std::decay_t<TCallable>, std::nullptr_t>) {

        // Allocate buffer
        buffer_allocate_options options{};
        options.flags = allocate_shared ? HOST_BUFFER_ALLOCATE_SHARED
                                        : HOST_BUFFER_ALLOCATE_CPU_ONLY;
        options.alignment = alignment;
        try_var(buffer, buffer_allocator::host().allocate(body_size, options));
        auto host_storage = buffer.as<host_buffer_t>().unwrap();
        buffer_storage = host_storage;
        // Read section into buffer
        {
            try_var(mapped, host_storage->map(map_write));
            if (!sr) {
                memcpy(mapped.buffer().data(), src_span.data(), body_size);
            } else {
                sr->read_span(mapped.buffer());
            }

            span =
                allocate_shared
                    ? std::span(reinterpret_cast<const std::byte *>(
                                    host_storage->physical_address().unwrap()),
                                body_size)
                    : mapped.buffer();
        }

        if (allocate_shared) {
            try_(host_storage->sync(sync_write_back));
        }

        return ok(span);

        // }
    } else {
        return callable(*sr, buffer_storage, body_size, alignment, allocate_shared);
    }
}

END_NS_NNCASE_RUNTIME
