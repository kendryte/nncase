/* Copyright 2020 Canaan Inc.
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
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_module.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

namespace
{
class runtime_module_init_context_impl : public runtime_module_init_context
{
public:
    runtime_module_init_context_impl(const module_header &header, interpreter &interp, gsl::span<const gsl::byte> sections) noexcept
        : header_(header), interp_(interp), sections_(sections)
    {
    }

    interpreter &interp() noexcept override
    {
        return interp_;
    }

    const module_header &header() noexcept override
    {
        return header_;
    }

    bool is_section_pinned() const noexcept override
    {
        return true;
    }

    gsl::span<const gsl::byte> section(const char *name) noexcept override
    {
        span_reader reader(sections_);
        while (!reader.empty())
        {
            auto header = reader.get_ref<section_header>();
            if (!strncmp(header->name, name, std::size(header->name)))
            {
                gsl::span<const gsl::byte> result;
                if (header->flags & SECTION_MERGED_INTO_RDATA)
                {
                    auto rdata_span = section(".rdata");
                    result = rdata_span.subspan(header->start, header->size);
                }
                else
                {
                    result = reader.read_avail().subspan(header->start, header->size);
                }

                return result;
            }
            else
            {
                if (!(header->flags & SECTION_MERGED_INTO_RDATA))
                    reader.skip((size_t)header->start + header->size);
            }
        }

        return {};
    }

private:
    const module_header &header_;
    interpreter &interp_;
    gsl::span<const gsl::byte> sections_;
};
}

const module_type_t &runtime_module::type() const noexcept
{
    return header_.type;
}

uint32_t runtime_module::mempools_size() const noexcept
{
    return header_.mempools;
}

const mempool_desc &runtime_module::mempool(size_t index) const noexcept
{
    assert(index < mempools_.size());
    return mempools_[index];
}

mempool_desc runtime_module::mempool(memory_location_t location) const noexcept
{
    for (auto &desc : mempools_)
    {
        if (desc.location == location)
            return desc;
    }

    mempool_desc desc {};
    desc.location = location;
    return desc;
}

uint32_t runtime_module::inputs_size() const noexcept
{
    return header_.inputs;
}

uint32_t runtime_module::outputs_size() const noexcept
{
    return header_.outputs;
}

const memory_range &runtime_module::input_desc(size_t index) const noexcept
{
    assert(index < input_tensors_.size());
    return input_tensors_[index].range;
}

const memory_range &runtime_module::output_desc(size_t index) const noexcept
{
    assert(index < output_tensors_.size());
    return output_tensors_[index].range;
}

const runtime_shape_t &runtime_module::input_shape(size_t index) const noexcept
{
    assert(index < input_tensors_.size());
    return input_tensors_[index].shape;
}

const runtime_shape_t &runtime_module::output_shape(size_t index) const noexcept
{
    assert(index < output_tensors_.size());
    return output_tensors_[index].shape;
}

result<void> runtime_module::initialize(const module_header &header, interpreter &interp) noexcept
{
    interp_ = &interp;
    header_ = header;
    span_reader reader(gsl::make_span(reinterpret_cast<const gsl::byte *>(&header) + sizeof(module_header), header.size));
    try
    {
        input_tensors_.resize(inputs_size());
        output_tensors_.resize(outputs_size());
        mempools_.resize(mempools_size());
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    // mempools
    for (auto &desc : mempools_)
        reader.read(desc);

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

    runtime_module_init_context_impl init_context(header, interp, reader.read_avail());
    return initialize_core(init_context);
}

#define INOUT_TENSOR_GETTER_IMPL(name)                              \
    if (index >= name##_tensors_.size())                            \
        return err(std::errc::result_out_of_range);                 \
                                                                    \
    auto &info = name##_tensors_[index];                            \
    if (info.bind_tensor.empty())                                   \
        try_set(info.bind_tensor, allocate_##name##_tensor(index)); \
    return ok(info.bind_tensor)

result<runtime_tensor> runtime_module::input_tensor(size_t index) noexcept
{
    INOUT_TENSOR_GETTER_IMPL(input);
}

result<runtime_tensor> runtime_module::output_tensor(size_t index) noexcept
{
    INOUT_TENSOR_GETTER_IMPL(output);
}

result<void> runtime_module::input_tensor(size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.empty())
        return err(std::errc::invalid_argument);
    if (index >= input_tensors_.size())
        return err(std::errc::result_out_of_range);

    auto &info = input_tensors_[index];
    if (info.range.datatype != tensor.datatype())
        return err(nncase_errc::datatype_mismatch);
    if (info.shape != tensor.shape())
        return err(nncase_errc::shape_mismatch);
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

result<void> runtime_module::output_tensor(size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.empty())
        return err(std::errc::invalid_argument);
    if (index >= output_tensors_.size())
        return err(std::errc::result_out_of_range);

    auto &info = output_tensors_[index];
    if (info.range.datatype != tensor.datatype())
        return err(nncase_errc::datatype_mismatch);
    if (info.shape != tensor.shape())
        return err(nncase_errc::shape_mismatch);
    if (info.bind_tensor != tensor)
    {
        if (validate_input_tensor(index, tensor).is_err())
        {
            auto device_tensor = info.device_tensor;
            if (device_tensor.empty())
                try_var(device_tensor, allocate_input_tensor(index));
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

result<void> runtime_module::run() noexcept
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
    try_(run_core());

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

result<void> runtime_module::initialize_inter_modules(interpreter &interp) noexcept
{
    return ok();
}
