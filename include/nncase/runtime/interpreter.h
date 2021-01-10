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
#pragma once
#include "allocator.h"
#include "model.h"
#include "result.h"
#include "runtime.h"
#include <gsl/gsl-lite.hpp>
#include <memory>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API interpreter
{
public:
    interpreter(host_allocator &host_allocator, allocation_state &alloc_state) noexcept;

    NNCASE_NODISCARD result<void> load_model(gsl::span<const gsl::byte> buffer) noexcept;

    size_t inputs_size() const noexcept { return input_ranges_.size(); }
    size_t outputs_size() const noexcept { return output_ranges_.size(); }
    memory_range input_range(size_t index) const { return input_ranges_[index]; }
    memory_range output_range(size_t index) const { return output_ranges_[index]; }
    runtime_shape_t input_shape(size_t index) const { return input_shapes_[index]; }
    runtime_shape_t output_shape(size_t index) const { return output_shapes_[index]; }
    gsl::span<gsl::byte> input_buffer(size_t index) const { return memory_at(input_range(index)); }
    gsl::span<gsl::byte> output_buffer(size_t index) const { return memory_at(output_range(index)); }

    gsl::span<gsl::byte> memory_at(memory_location_t location) const noexcept;
    gsl::span<gsl::byte> section_memory_at(gsl::zstring_span section_name) const noexcept;

    template <class T>
    gsl::span<T> memory_at(const memory_range &range) const noexcept
    {
        auto span = memory_at(range);
        return { reinterpret_cast<T *>(span.data()), span.size() / sizeof(T) };
    }

    result<void> run();

protected:
    NNCASE_NODISCARD result<void> initialize_target() noexcept;
    gsl::span<gsl::byte> memory_at(const memory_range &range) const noexcept;

private:
    void set_memory(memory_location_t location, gsl::span<gsl::byte> buffer);

private:
    host_allocator &host_allocator_;
    allocation_state &alloc_state_;
    const model_header *model_header_;
    gsl::span<const memory_range> input_ranges_;
    gsl::span<const memory_range> output_ranges_;
    gsl::span<const runtime_shape_t> input_shapes_;
    gsl::span<const runtime_shape_t> output_shapes_;
    gsl::span<const section_desc> section_descs_;
    std::vector<gsl::span<gsl::byte>> section_mems_;
    gsl::span<gsl::byte> text_section_;
    std::array<gsl::span<gsl::byte>, 4> memory_locations_;
    std::unique_ptr<runtime_base> runtime_;
};

END_NS_NNCASE_RUNTIME
