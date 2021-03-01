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
#include "runtime_module.h"
#include <nncase/runtime/k210/error.h>
#include <nncase/runtime/runtime_loader.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

result<void> k210_runtime_module::initialize_core(runtime_module_init_context &context) noexcept
{
    assert(context.is_section_pinned());
    auto data_pool = mempool(mem_data);
    if (data_pool.size)
    {
        data_.reset(new (std::nothrow) gsl::byte[data_pool.size]);
        if (!data_)
            return err(std::errc::not_enough_memory);
    }

    rdata_ = context.section(".rdata");
    text_ = context.section(".text");
    return ok();
}

result<runtime_tensor> k210_runtime_module::allocate_input_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(input_desc(index).datatype, input_shape(index));
}

result<runtime_tensor> k210_runtime_module::allocate_output_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(output_desc(index).datatype, output_shape(index));
}

result<void> k210_runtime_module::validate_input_tensor(size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> k210_runtime_module::validate_output_tensor(size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> k210_runtime_module::run_core() noexcept
{
    return visit(text_);
}

result<gsl::span<gsl::byte>> k210_runtime_module::memory_at(const memory_range &mrange) noexcept
{
    gsl::byte *base;
    switch (mrange.memory_location)
    {
    case mem_input:
    {
        size_t id = -1;
        for (size_t i = 0; i < inputs_size(); i++)
        {
            if (mrange.start == input_desc(i).start)
            {
                id = i;
                break;
            }
        }

        if (id != -1)
        {
            try_var(tensor, input_tensor(id));
            try_var(buffer, host_runtime_tensor::buffer(tensor));
            base = buffer.data() - mrange.start;
        }
        else
        {
            return err(std::errc::invalid_argument);
        }
        break;
    }
    case mem_output:
    {
        size_t id = -1;
        for (size_t i = 0; i < outputs_size(); i++)
        {
            if (mrange.start == output_desc(i).start)
            {
                id = i;
                break;
            }
        }

        if (id != -1)
        {
            try_var(tensor, output_tensor(id));
            try_var(buffer, host_runtime_tensor::buffer(tensor));
            base = buffer.data() - mrange.start;
        }
        else
        {
            return err(std::errc::invalid_argument);
        }
        break;
    }
    case mem_rdata:
        base = const_cast<gsl::byte *>(rdata_.data());
        break;
    case mem_data:
        base = data_.get();
        break;
    case mem_kpu:
#ifdef NNCASE_SIMULATOR
        base = kpu_ram_.data();
#endif
        break;
    default:
        return err(nncase_errc::invalid_memory_location);
    }

    return ok(gsl::make_span(base + mrange.start, mrange.size));
}

result<std::unique_ptr<runtime_module>> k210::create_k210_runtime_module()
{
    std::unique_ptr<runtime_module> mod(new (std::nothrow) k210_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

extern "C"
{
    NNCASE_MODULES_K210_API void RUNTIME_MODULE_ACTIVATOR_NAME(result<std::unique_ptr<runtime_module>> &result)
    {
        result = create_k210_runtime_module();
    }
}
