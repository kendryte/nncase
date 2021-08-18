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
#include <nncase/runtime/runtime_function.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

namespace
{
class runtime_function_init_context_impl : public runtime_function_init_context
{
public:
    runtime_function_init_context_impl(const function_header &header, runtime_module_init_context &module_init_context, gsl::span<const gsl::byte> body) noexcept
        : header_(header), module_init_context_(module_init_context), body_(body)
    {
    }

    runtime_module_init_context &module_init_context() noexcept override
    {
        return module_init_context_;
    }

    const function_header &header() noexcept override
    {
        return header_;
    }

    gsl::span<const gsl::byte> body() noexcept override
    {
        return body_;
    }

private:
    const function_header &header_;
    runtime_module_init_context &module_init_context_;
    gsl::span<const gsl::byte> body_;
};
}

runtime_function::runtime_function(runtime_module &rt_module)
    : rt_module_(rt_module)
{
}

runtime_module &runtime_function::module() const noexcept
{
    return rt_module_;
}

uint32_t runtime_function::inputs_size() const noexcept
{
    return header_.inputs;
}

uint32_t runtime_function::outputs_size() const noexcept
{
    return header_.outputs;
}

const memory_range &runtime_function::input_desc(size_t index) const noexcept
{
    assert(index < input_tensors_.size());
    return input_tensors_[index].range;
}

const memory_range &runtime_function::output_desc(size_t index) const noexcept
{
    assert(index < output_tensors_.size());
    return output_tensors_[index].range;
}

const runtime_shape_t &runtime_function::input_shape(size_t index) const noexcept
{
    assert(index < input_tensors_.size());
    return input_tensors_[index].shape;
}

const runtime_shape_t &runtime_function::output_shape(size_t index) const noexcept
{
    assert(index < output_tensors_.size());
    return output_tensors_[index].shape;
}

result<void> runtime_function::initialize(gsl::span<const gsl::byte> payload, runtime_module_init_context &module_init_context) noexcept
{
    span_reader reader(payload);
    reader.read(header_);

    try
    {
        input_tensors_.resize(inputs_size());
        output_tensors_.resize(outputs_size());
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    auto read_shape = [&](runtime_shape_t &shape) {
        shape.resize(reader.read<uint32_t>());
        for (auto &dim : shape)
            dim = reader.read<uint32_t>();
    };

    // inputs
    for (auto &in : input_tensors_)
        reader.read(in.range);
    for (auto &in : input_tensors_)
        read_shape(in.shape);

    // outputs
    for (auto &out : output_tensors_)
        reader.read(out.range);
    for (auto &out : output_tensors_)
        read_shape(out.shape);

    runtime_function_init_context_impl init_context(header_, module_init_context, reader.read_avail());
    return initialize_core(init_context);
}

#define INOUT_TENSOR_GETTER_IMPL(name)                                              \
    CHECK_WITH_ERR(index < name##_tensors_.size(), std::errc::result_out_of_range); \
                                                                                    \
    auto &info = name##_tensors_[index];                                            \
    if (info.bind_tensor.empty())                                                   \
    {                                                                               \
        try_set(info.bind_tensor, allocate_##name##_tensor(index));                 \
    }                                                                               \
    return ok(info.bind_tensor);

result<runtime_tensor> runtime_function::input_tensor(size_t index) noexcept
{
    INOUT_TENSOR_GETTER_IMPL(input);
}

result<runtime_tensor> runtime_function::output_tensor(size_t index) noexcept
{
    INOUT_TENSOR_GETTER_IMPL(output);
}

#define DEV_INOUT_TENSOR_GETTER_IMPL(name)                                          \
    CHECK_WITH_ERR(index < name##_tensors_.size(), std::errc::result_out_of_range); \
                                                                                    \
    auto &info = name##_tensors_[index];                                            \
    if (info.bind_tensor.empty())                                                   \
    {                                                                               \
        try_set(info.bind_tensor, allocate_##name##_tensor(index));                 \
    }                                                                               \
    if (!info.device_tensor.empty())                                                \
    {                                                                               \
        return ok(info.device_tensor);                                              \
    }                                                                               \
    return ok(info.bind_tensor);

result<runtime_tensor> runtime_function::device_input_tensor(size_t index) noexcept
{
    DEV_INOUT_TENSOR_GETTER_IMPL(input);
}

result<runtime_tensor> runtime_function::device_output_tensor(size_t index) noexcept
{
    DEV_INOUT_TENSOR_GETTER_IMPL(output);
}

result<void> runtime_function::input_tensor(size_t index, runtime_tensor tensor) noexcept
{
    CHECK_WITH_ERR(!tensor.empty(), std::errc::invalid_argument);
    CHECK_WITH_ERR(index < input_tensors_.size(), std::errc::result_out_of_range);

    auto &info = input_tensors_[index];
    CHECK_WITH_ERR(info.range.datatype == tensor.datatype(), nncase_errc::datatype_mismatch);
    CHECK_WITH_ERR(info.shape == tensor.shape(), nncase_errc::shape_mismatch);

    if (info.bind_tensor != tensor)
    {
        if (validate_input_tensor(index, tensor).is_err())
        {
            auto device_tensor = info.device_tensor;
            if (device_tensor.empty())
                try_var(device_tensor, allocate_input_tensor(index));
            if (!tensor.can_copy_to_without_staging(device_tensor))
            {
                try_set(info.staging_tensor, host_runtime_tensor::create(info.range.datatype, info.shape));
            }
            else
            {
                info.staging_tensor.reset();
            }

            info.device_tensor = device_tensor;
        }
        else
        {
            info.device_tensor.reset();
            info.staging_tensor.reset();
        }

        info.bind_tensor = tensor;
    }

    return ok();
}

result<void> runtime_function::output_tensor(size_t index, runtime_tensor tensor) noexcept
{
    CHECK_WITH_ERR(!tensor.empty(), std::errc::invalid_argument);
    CHECK_WITH_ERR(index < output_tensors_.size(), std::errc::result_out_of_range);

    auto &info = output_tensors_[index];
    CHECK_WITH_ERR(info.range.datatype == tensor.datatype(), nncase_errc::datatype_mismatch);
    CHECK_WITH_ERR(info.shape == tensor.shape(), nncase_errc::shape_mismatch);

    if (info.bind_tensor != tensor)
    {
        if (validate_output_tensor(index, tensor).is_err())
        {
            auto device_tensor = info.device_tensor;
            if (device_tensor.empty())
                try_var(device_tensor, allocate_output_tensor(index));
            if (!device_tensor.can_copy_to_without_staging(tensor))
            {
                try_set(info.staging_tensor, host_runtime_tensor::create(info.range.datatype, info.shape));
            }
            else
            {
                info.staging_tensor.reset();
            }

            info.device_tensor = device_tensor;
        }
        else
        {
            info.device_tensor.reset();
            info.staging_tensor.reset();
        }

        info.bind_tensor = tensor;
    }

    return ok();
}

result<void> runtime_function::invoke() noexcept
{
    // 1. Ensure bindings
    for (size_t i = 0; i < input_tensors_.size(); i++)
    {
        auto &info = input_tensors_[i];
        if (info.bind_tensor.empty())
            try_set(info.bind_tensor, allocate_input_tensor(i));
    }

    for (size_t i = 0; i < output_tensors_.size(); i++)
    {
        auto &info = output_tensors_[i];
        if (info.bind_tensor.empty())
            try_set(info.bind_tensor, allocate_output_tensor(i));
    }

    // 2. Copy inputs
    for (auto &in : input_tensors_)
    {
        if (in.staging_tensor.empty())
        {
            if (!in.device_tensor.empty())
                try_(in.bind_tensor.copy_to(in.device_tensor));
        }
        else
        {
            try_(in.bind_tensor.copy_to(in.staging_tensor));
            try_(in.staging_tensor.copy_to(in.device_tensor));
        }
    }

    // 3. Run
    try_(invoke_core());

    // 4. Copy outputs
    for (auto &out : output_tensors_)
    {
        if (out.staging_tensor.empty())
        {
            if (!out.device_tensor.empty())
                try_(out.device_tensor.copy_to(out.bind_tensor));
        }
        else
        {
            try_(out.device_tensor.copy_to(out.staging_tensor));
            try_(out.staging_tensor.copy_to(out.bind_tensor));
        }
    }

    return ok();
}
