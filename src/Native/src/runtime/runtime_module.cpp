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

    gsl::span<const gsl::byte> section(const char *name) noexcept override {
        return find_section(name, sections_);
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
} // namespace

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

    span_reader func_reader(read_functions(reader, header_.functions));
    runtime_module_init_context_impl init_context(
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
