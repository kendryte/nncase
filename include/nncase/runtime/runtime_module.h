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
#include "model.h"
#include "result.h"

BEGIN_NS_NNCASE_RUNTIME

class interpreter;

class NNCASE_API runtime_module
{
public:
    static result<std::unique_ptr<runtime_module>> create(const module_header &header);

    runtime_module(const module_header &header) noexcept;
    runtime_module(runtime_module &) = delete;
    virtual ~runtime_module() = default;

    result<void> initialize(interpreter &interp) noexcept;
    const module_type_t &type() const noexcept;

    uint32_t mempools_count() const noexcept;
    const mempool_desc &mempool_desc(size_t index) const noexcept;

    uint32_t inputs_count() const noexcept;
    const shape_header &input_shape(size_t index) const noexcept;
    const memory_range &input_desc(size_t index) const noexcept;
    gsl::span<gsl::byte> input_buffer(size_t index) const noexcept;

    uint32_t outputs_count() const noexcept;
    const shape_header &output_shape(size_t index) const noexcept;
    gsl::span<gsl::byte> output_buffer(size_t index) const noexcept;

    virtual result<void> execute() noexcept = 0;

protected:
    virtual result<void> initialize_core(interpreter &interp) noexcept = 0;

private:
    const module_header &header_;
};

END_NS_NNCASE_RUNTIME
