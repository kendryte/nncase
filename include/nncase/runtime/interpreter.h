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
#include "runtime_module.h"
#include <gsl/gsl-lite.hpp>
#include <memory>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API interpreter
{
public:
    interpreter() noexcept;
    interpreter(interpreter &) = delete;
    interpreter(interpreter &&) = default;

    NNCASE_NODISCARD result<void> load_model(gsl::span<const gsl::byte> buffer) noexcept;

    size_t inputs_size() const noexcept;
    size_t outputs_size() const noexcept;
    memory_range input_range(size_t index) const;
    memory_range output_range(size_t index) const;
    const shape_header &input_shape(size_t index) const;
    const shape_header &output_shape(size_t index) const;
    gsl::span<gsl::byte> input_buffer(size_t index);
    gsl::span<gsl::byte> output_buffer(size_t index);

    result<void> run();

private:
    void set_memory(memory_location_t location, gsl::span<gsl::byte> buffer);

private:
    std::vector<std::unique_ptr<runtime_module>> modules_;
    runtime_module *main_module_;
};

END_NS_NNCASE_RUNTIME
