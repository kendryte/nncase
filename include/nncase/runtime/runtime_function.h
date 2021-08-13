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
class runtime_module;
struct runtime_module_init_context;

struct NNCASE_API runtime_function_init_context
{
    virtual runtime_module_init_context &module_init_context() noexcept = 0;
    virtual const function_header &header() noexcept = 0;
    virtual gsl::span<const gsl::byte> body() noexcept = 0;
};

class NNCASE_API runtime_function
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
    runtime_function(runtime_module &rt_module);
    runtime_function(const runtime_function &) = delete;
    virtual ~runtime_function() = default;
    runtime_function &operator=(const runtime_function &) = delete;

    result<void> initialize(gsl::span<const gsl::byte> payload, runtime_module_init_context &module_init_context) noexcept;
    runtime_module &module() const noexcept;

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

    result<void> invoke() noexcept;

protected:
    virtual result<void> initialize_core(runtime_function_init_context &context) noexcept = 0;
    virtual result<runtime_tensor> allocate_input_tensor(size_t index) noexcept = 0;
    virtual result<runtime_tensor> allocate_output_tensor(size_t index) noexcept = 0;
    virtual result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept = 0;
    virtual result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept = 0;
    result<runtime_tensor> device_input_tensor(size_t index) noexcept;
    result<runtime_tensor> device_output_tensor(size_t index) noexcept;
    virtual result<void> invoke_core() noexcept = 0;

private:
    function_header header_;
    std::vector<inout_tensor_info> input_tensors_;
    std::vector<inout_tensor_info> output_tensors_;
    runtime_module &rt_module_;
};

END_NS_NNCASE_RUNTIME
