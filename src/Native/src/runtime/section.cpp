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
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

gsl::span<const gsl::byte>
runtime::find_section(const char *name,
                      gsl::span<const gsl::byte> sections) noexcept {
    span_reader reader(sections);
    while (!reader.empty()) {
        auto header = reader.peek_ref<section_header>();
        if (!strncmp(header->name, name, MAX_SECTION_NAME_LENGTH)) {
            gsl::span<const gsl::byte> result;
            if (header->flags & SECTION_MERGED_INTO_RDATA) {
                auto rdata_span = find_section(".rdata", sections);
                result =
                    rdata_span.subspan(header->body_start, header->body_size);
            } else {
                reader.skip(sizeof(section_header));
                result = reader.read_avail().subspan(header->body_start,
                                                     header->body_size);
            }

            return result;
        } else {
            reader.skip(header->size);
        }
    }

    return {};
}

gsl::span<const gsl::byte> runtime::read_sections(span_reader &sr,
                                                  size_t sections) noexcept {
    auto nest_sr = sr;
    size_t size = 0;

    for (size_t i = 0; i < sections; i++) {
        auto header = nest_sr.peek_ref<section_header>();
        size += header->size;
        nest_sr.skip(header->size);
    }

    return sr.read_span(size);
}

result<std::streampos> runtime::find_section(const char *name,
                                             stream_reader &reader,
                                             size_t max_sections) noexcept {
    const auto sections_pos = reader.tell();
    for (size_t i = 0; i < max_sections; i++) {
        auto pos = reader.tell();
        auto header = reader.read<section_header>();
        if (!strncmp(header.name, name, MAX_SECTION_NAME_LENGTH)) {
            std::streampos body_pos;
            if (header.flags & SECTION_MERGED_INTO_RDATA) {
                reader.seek(sections_pos);
                try_var(rdata_pos,
                        find_section(".rdata", reader, max_sections));
                body_pos = rdata_pos + std::streampos(header.body_start);
            } else {
                body_pos =
                    pos + std::streampos(sizeof(header) + header.body_start);
            }

            reader.seek(pos);
            return ok(body_pos);
        } else {
            reader.skip(header.size - sizeof(header));
        }
    }

    return err(std::errc::no_such_file_or_directory);
}
