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
#include "host_buffer.h"
#include "model.h"
#include "result.h"
#include "span_reader.h"
#include "stream_reader.h"
#include <nncase/type.h>
#include <nncase/value.h>

BEGIN_NS_NNCASE_RUNTIME

struct NNCASE_API runtime_section_context {
    virtual bool is_section_pinned() const noexcept = 0;
    virtual result<gsl::span<const gsl::byte>>
    section(const char *name) noexcept = 0;
    virtual result<stream_reader *>
    seek_section(const char *name, section_header &header) noexcept = 0;

    result<gsl::span<const gsl::byte>>
    get_or_read_section(const char *name, host_buffer_t &storage,
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

END_NS_NNCASE_RUNTIME
