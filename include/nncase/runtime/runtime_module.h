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
#include "model.h"
#include "result.h"
#include "runtime_tensor.h"

BEGIN_NS_NNCASE_RUNTIME

class interpreter;

struct NNCASE_API runtime_module_init_context
{
    virtual bool is_section_pinned() const noexcept = 0;
    virtual interpreter &interp() noexcept = 0;
    virtual const module_header &header() noexcept = 0;
    virtual gsl::span<const gsl::byte> section(const char *name) noexcept = 0;
};

class NNCASE_API runtime_module
{
private:
    struct inout_tensor_info
    {
        runtime_shape_t shape;
        runtime_shape_t strides;
        memory_range range;
        runtime_tensor bind_tensor;
        runtime_tensor staging_tensor;
        runtime_tensor device_tensor;
    };

public:
    static result<std::unique_ptr<runtime_module>> create(const module_type_t &type);

    runtime_module() = default;
    runtime_module(runtime_module &) = delete;
    virtual ~runtime_module() = default;

    result<void> initialize(const module_header &header, interpreter &interp) noexcept;
    virtual result<void> initialize_inter_modules(interpreter &interp) noexcept;
    const module_type_t &type() const noexcept;

    interpreter &interp() const noexcept { return *interp_; }

    uint32_t mempools_size() const noexcept;
    const mempool_desc &mempool(size_t index) const noexcept;
    mempool_desc mempool(memory_location_t location) const noexcept;

    uint32_t inputs_size() const noexcept;
    const runtime_shape_t &input_shape(size_t index) const noexcept;
    const memory_range &input_desc(size_t index) const noexcept;
    result<runtime_tensor> input_tensor(size_t index) noexcept;
    result<void> input_tensor(size_t index, runtime_tensor tensor) noexcept;

    uint32_t outputs_size() const noexcept;
    const runtime_shape_t &output_shape(size_t index) const noexcept;
    const memory_range &output_desc(size_t index) const noexcept;
    result<runtime_tensor> output_tensor(size_t index) noexcept;
    result<void> output_tensor(size_t index, runtime_tensor tensor) noexcept;

    result<void> run() noexcept;

protected:
    virtual result<void> initialize_core(runtime_module_init_context &context) noexcept = 0;
    virtual result<runtime_tensor> allocate_input_tensor(size_t index) noexcept = 0;
    virtual result<runtime_tensor> allocate_output_tensor(size_t index) noexcept = 0;
    virtual result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept = 0;
    virtual result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept = 0;
    result<runtime_tensor> device_input_tensor(size_t index) noexcept;
    result<runtime_tensor> device_output_tensor(size_t index) noexcept;
    virtual result<void> run_core() noexcept = 0;

private:
    module_header header_;
    std::vector<mempool_desc> mempools_;
    std::vector<inout_tensor_info> input_tensors_;
    std::vector<inout_tensor_info> output_tensors_;
    interpreter *interp_ = nullptr;
};

END_NS_NNCASE_RUNTIME
