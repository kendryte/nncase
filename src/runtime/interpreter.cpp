/* Copyright 2019-2020 Canaan Inc.
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
#include "runtime_loader.h"
#include <cassert>
#include <iostream>
#include <nncase/runtime/error.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

interpreter::interpreter(host_allocator &host_allocator, allocation_state &alloc_state) noexcept
    : host_allocator_(host_allocator), alloc_state_(alloc_state), model_header_(nullptr)
{
}

result<void> interpreter::load_model(gsl::span<const gsl::byte> buffer) noexcept
{
    // 1. Analyze kmodel
    span_reader reader(buffer);
    reader.get_ref(model_header_);
    // Validate model
    if (model_header_->identifier != MODEL_IDENTIFIER)
        return make_error_condition(nncase_errc::invalid_model_indentifier);
    if (model_header_->version != MODEL_VERSION)
        return make_error_condition(nncase_errc::invalid_model_version);

    input_ranges_ = reader.read_span<memory_range>(model_header_->inputs);
    input_shapes_ = reader.read_span<runtime_shape_t>(model_header_->inputs);
    output_ranges_ = reader.read_span<memory_range>(model_header_->outputs);
    output_shapes_ = reader.read_span<runtime_shape_t>(model_header_->outputs);

#ifndef NDEBUG
    for (size_t i = 0; i < input_ranges_.size(); ++i)
        printf("Input %d @%d, size=%d\n", (int)i, input_ranges_[i].start, input_ranges_[i].size);

    for (size_t i = 0; i < output_ranges_.size(); ++i)
        printf("Output %d @%d, size=%d\n", (int)i, output_ranges_[i].start, output_ranges_[i].size);
#endif

    section_descs_ = reader.read_span<section_desc>(model_header_->sections);
    // 2. Load sections
    for (auto &sec : section_descs_)
    {
        auto sec_buffer = gsl::make_span(const_cast<gsl::byte *>(buffer.data()) + sec.offset, sec.size_in_file);
        if (sec.size > sec.size_in_file)
        {
            auto new_buffer = host_allocator_.allocate(alloc_state_, sec.size);
            if (new_buffer.empty())
                return make_error_condition(std::errc::not_enough_memory);
            std::memcpy(new_buffer.data(), sec_buffer.data(), sec_buffer.size());
            sec_buffer = new_buffer;
            section_mems_.emplace_back(new_buffer);
        }

        // pc
        if (!strcmp(sec.name, ".text"))
            text_section_ = sec_buffer;
        // input
        else if (!strcmp(sec.name, ".input"))
            set_memory(mem_input, sec_buffer);
        // output
        else if (!strcmp(sec.name, ".output"))
            set_memory(mem_output, sec_buffer);
        // rdata
        else if (!strcmp(sec.name, ".rdata"))
            set_memory(mem_rdata, sec_buffer);
        // data
        else if (!strcmp(sec.name, ".data"))
            set_memory(mem_data, sec_buffer);

#ifndef NDEBUG
        printf("Load section %s @%p\n", sec.name, sec_buffer.data());
#endif
    }

    try_(initialize_target());
    return ok();
}

gsl::span<gsl::byte> interpreter::memory_at(const memory_range &range) const noexcept
{
    auto base = memory_at(range.memory_location);
    return base.subspan(range.start, range.size);
}

gsl::span<gsl::byte> interpreter::memory_at(memory_location_t location) const noexcept
{
    return memory_locations_[(size_t)location];
}

gsl::span<gsl::byte> interpreter::section_memory_at(gsl::zstring_span section_name) const noexcept
{
    size_t index = 0;
    for (auto &sec : section_descs_)
    {
        if (!strcmp(sec.name, section_name.assume_z()))
            return section_mems_[index];
        index++;
    }

    return {};
}

void interpreter::set_memory(memory_location_t location, gsl::span<gsl::byte> buffer)
{
    memory_locations_[(size_t)location] = buffer;
}

result<void> interpreter::initialize_target() noexcept
{
    result<std::unique_ptr<runtime_base>> result(err(std::errc::resource_unavailable_try_again));
    create_runtime(model_header_->target, result);
    if (result.is_ok())
    {
        runtime_ = std::move(result.unwrap());
        return ok();
    }
    else
    {
        return result.unwrap_err();
    }
}
