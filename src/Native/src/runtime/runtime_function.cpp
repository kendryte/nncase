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
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_function.h>
#include <nncase/runtime/span_reader.h>
#include <nncase/runtime/stackvm/op_profile.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
class runtime_function_init_context_span_impl
    : public runtime_function_init_context {
  public:
    runtime_function_init_context_span_impl(
        const function_header &header,
        runtime_module_init_context &module_init_context,
        gsl::span<const gsl::byte> sections) noexcept
        : header_(header),
          module_init_context_(module_init_context),
          sections_(sections) {}

    runtime_module_init_context &module_init_context() noexcept override {
        return module_init_context_;
    }

    const function_header &header() noexcept override { return header_; }

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
    const function_header &header_;
    runtime_module_init_context &module_init_context_;
    gsl::span<const gsl::byte> sections_;
};

class runtime_function_init_context_stream_impl
    : public runtime_function_init_context {
  public:
    runtime_function_init_context_stream_impl(
        const function_header &header,
        runtime_module_init_context &module_init_context, stream_reader &reader,
        std::streampos sections) noexcept
        : header_(header),
          module_init_context_(module_init_context),
          reader_(reader),
          sections_(sections) {}

    runtime_module_init_context &module_init_context() noexcept override {
        return module_init_context_;
    }

    const function_header &header() noexcept override { return header_; }

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
    const function_header &header_;
    runtime_module_init_context &module_init_context_;
    stream_reader &reader_;
    std::streampos sections_;
};
} // namespace

runtime_function::runtime_function(runtime_module &rt_module)
    : rt_module_(rt_module), return_type_(nullptr) {}

runtime_module &runtime_function::module() const noexcept { return rt_module_; }

uint32_t runtime_function::parameters_size() const noexcept {
    return header_.parameters;
}

result<type> runtime_function::parameter_type(size_t index) const noexcept {
    assert(index < parameter_types_.size());
    return ok(parameter_types_[index]);
}

const type &runtime_function::return_type() const noexcept {
    return return_type_;
}

result<void> runtime_function::initialize(
    gsl::span<const gsl::byte> payload,
    runtime_module_init_context &module_init_context) noexcept {
    span_reader reader(payload);
    reader.read(header_);

    try {
        parameter_types_.resize(parameters_size(), nullptr);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    // parameters
    for (auto &param : parameter_types_) {
        checked_try_set(param, deserialize_type(reader));
    }

    // return type
    checked_try_set(return_type_, deserialize_type(reader));

    runtime_function_init_context_span_impl init_context(
        header_, module_init_context, read_sections(reader, header_.sections));
    return initialize_core(init_context);
}

result<void> runtime_function::initialize(
    stream_reader &reader,
    runtime_module_init_context &module_init_context) noexcept {
    reader.read(header_);

    try {
        parameter_types_.resize(parameters_size(), nullptr);
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    // parameters
    for (auto &param : parameter_types_) {
        checked_try_set(param, deserialize_type(reader));
    }

    // return type
    checked_try_set(return_type_, deserialize_type(reader));

    const auto sections_pos = reader.tell();
    runtime_function_init_context_stream_impl init_context(
        header_, module_init_context, reader, sections_pos);
    return initialize_core(init_context);
}

result<value_t> runtime_function::invoke(gsl::span<value_t> parameters,
                                         value_t return_value) noexcept {
    checked_try_var(retval, invoke_core(parameters, return_value));
    try_var(enable_profiling,
            module().interp().options().get_scalar_opt<uint8_t>("profiling"));
    if (enable_profiling) {
        try_var(entry_func, module().interp().entry_function());
        if (entry_func == this) {
            op_profile::print();
        }
    }
    return ok(retval);
}
