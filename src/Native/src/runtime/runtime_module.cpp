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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_module.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
class runtime_module_init_context_span_impl
    : public runtime_module_init_context {
  public:
    runtime_module_init_context_span_impl(
        const module_header &header, interpreter &interp,
        gsl::span<const gsl::byte> sections) noexcept
        : header_(header), interp_(interp), sections_(sections) {}

    interpreter &interp() noexcept override { return interp_; }

    const module_header &header() noexcept override { return header_; }

    bool is_section_pinned() const noexcept override { return true; }

    result<gsl::span<const gsl::byte>>
    section(const char *name) noexcept override {
        return ok(find_section(name, sections_));
    }

    result<stream_reader *>
    seek_section([[maybe_unused]] const char *name,
                 [[maybe_unused]] section_header &header) noexcept override {
        return err(std::errc::not_supported);
    }

  private:
    const module_header &header_;
    interpreter &interp_;
    gsl::span<const gsl::byte> sections_;
};

gsl::span<const gsl::byte> read_functions(span_reader &sr,
                                          size_t functions) noexcept {
    auto nest_sr = sr;
    size_t size = 0;

    for (size_t i = 0; i < functions; i++) {
        auto func_size =
            nest_sr.peek_with_offset<decltype(function_header::size)>(
                offsetof(function_header, size));
        nest_sr.skip(func_size);
        size += func_size;
    }

    return sr.read_span(size);
}

class runtime_module_init_context_stream_impl
    : public runtime_module_init_context {
  public:
    runtime_module_init_context_stream_impl(const module_header &header,
                                            interpreter &interp,
                                            stream_reader &reader,
                                            std::streampos sections) noexcept
        : header_(header),
          interp_(interp),
          reader_(reader),
          sections_(sections) {}

    interpreter &interp() noexcept override { return interp_; }

    const module_header &header() noexcept override { return header_; }

    bool is_section_pinned() const noexcept override { return false; }

    result<gsl::span<const gsl::byte>>
    section([[maybe_unused]] const char *name) noexcept override {
        return err(std::errc::not_supported);
    }

    result<stream_reader *>
    seek_section(const char *name, section_header &header) noexcept override {
        reader_.seek(sections_);
        try_var(body_pos, find_section(name, reader_, header_.sections));
        reader_.read(header);
        reader_.seek(body_pos);
        return ok(&reader_);
    }

  private:
    const module_header &header_;
    interpreter &interp_;
    stream_reader &reader_;
    std::streampos sections_;
};

void skip_functions(stream_reader &sr, size_t functions) noexcept {
    for (size_t i = 0; i < functions; i++) {
        auto header = sr.read<function_header>();
        sr.skip(header.size - sizeof(header));
    }
}
} // namespace

result<gsl::span<const gsl::byte>>
runtime_module_init_context::get_or_read_section(
    const char *name, std::unique_ptr<gsl::byte[]> &storage) noexcept {
    gsl::span<const gsl::byte> span;

    auto section_span_r = section(name);
    if (section_span_r.is_ok()) {
        span = std::move(section_span_r).unwrap();
    } else {
        section_header header;
        try_var(sr, seek_section(name, header));
        storage = std::make_unique<gsl::byte[]>(header.body_size);
        gsl::span<gsl::byte> storage_span(storage.get(), header.body_size);
        sr->read_span(storage_span);
        span = storage_span;
    }

    return ok(span);
}

const module_kind_t &runtime_module::kind() const noexcept {
    return header_.kind;
}

result<void> runtime_module::initialize(gsl::span<const gsl::byte> payload,
                                        interpreter &interp) noexcept {
    interp_ = &interp;
    span_reader reader(payload);
    reader.read(header_);

    try {
        functions_.resize(header_.functions);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    span_reader func_reader(read_functions(reader, header_.functions));
    runtime_module_init_context_span_impl init_context(
        header_, interp, read_sections(reader, header_.sections));
    try_(initialize_before_functions(init_context));

    for (size_t i = 0; i < header_.functions; i++) {
        auto func_size =
            func_reader.peek_with_offset<decltype(function_header::size)>(
                offsetof(function_header, size));
        auto payload = func_reader.read_span(func_size);
        try_var(func, create_function());
        try_(func->initialize(payload, init_context));
        functions_[i] = std::move(func);
    }

    return initialize_after_functions(init_context);
}

result<void> runtime_module::initialize(stream_reader &reader,
                                        interpreter &interp) noexcept {
    interp_ = &interp;
    reader.read(header_);

    try {
        functions_.resize(header_.functions);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    const auto functions_pos = reader.tell();
    skip_functions(reader, header_.functions);
    const auto sections_pos = reader.tell();
    runtime_module_init_context_stream_impl init_context(header_, interp,
                                                         reader, sections_pos);
    try_(initialize_before_functions(init_context));

    auto cnt_functions_pos = functions_pos;
    for (size_t i = 0; i < header_.functions; i++) {
        reader.seek(cnt_functions_pos);
        auto func_header = reader.read<function_header>();
        try_var(func, create_function());
        reader.seek(cnt_functions_pos);
        try_(func->initialize(reader, init_context));
        functions_[i] = std::move(func);
        cnt_functions_pos += func_header.size;
    }

    return initialize_after_functions(init_context);
}

result<runtime_function *>
runtime_module::find_function_by_id(size_t index) noexcept {
    CHECK_WITH_ERR(index < functions_.size(), std::errc::result_out_of_range);
    return ok(functions_[index].get());
}

result<void> runtime_module::initialize_before_functions(
    NNCASE_UNUSED runtime_module_init_context &context) noexcept {
    return ok();
}

result<void> runtime_module::initialize_after_functions(
    NNCASE_UNUSED runtime_module_init_context &context) noexcept {
    return ok();
}
