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
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

stackvm_runtime_module &stackvm_runtime_function::module() const noexcept
{
    return static_cast<stackvm_runtime_module &>(runtime_function::module());
}

result<void> stackvm_runtime_function::initialize_core(runtime_function_init_context &context) noexcept
{
    text_ = context.module_init_context().section(".text").subspan(context.header().entrypoint, context.header().text_size);
    return ok();
}

result<runtime_tensor> stackvm_runtime_function::allocate_input_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(input_desc(index).datatype, input_shape(index));
}

result<runtime_tensor> stackvm_runtime_function::allocate_output_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(output_desc(index).datatype, output_shape(index));
}

result<void> stackvm_runtime_function::validate_input_tensor(NNCASE_UNUSED size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> stackvm_runtime_function::validate_output_tensor(NNCASE_UNUSED size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host() && tensor.is_contiguous())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> stackvm_runtime_function::invoke_core() noexcept
{
    call_depth_ = 0;
    return visit(text_);
}

uintptr_t stackvm_runtime_function::pc() const noexcept
{
    return (uintptr_t)(text_.size_bytes() - reader_.avail());
}

result<void> stackvm_runtime_function::pc(uintptr_t value) noexcept
{
    if (value >= text_.size_bytes())
        return err(nncase_errc::stackvm_illegal_target);
    reader_ = span_reader(text_.subspan(value));
    return ok();
}

result<void> stackvm_runtime_function::pc_relative(intptr_t offset) noexcept
{
    return pc((uintptr_t)((intptr_t)pc() + offset));
}

result<padding> stackvm_runtime_function::pop_padding() noexcept
{
    try_var(interior, stack_.pop());
    try_var(after, stack_.pop());
    try_var(before, stack_.pop());
    return ok(padding { before.as_i4(), after.as_i4(), interior.as_i4() });
}

result<uintptr_t> stackvm_runtime_function::pop_addr() noexcept
{
    try_var(addr, stack_.pop());
    return ok(addr.as_u());
}

runtime_axis_t stackvm_runtime_function::as_runtime_axis(const runtime_shape_t &shape)
{
    runtime_axis_t axis(shape.size());
    for (size_t i = 0; i < shape.size(); i++)
        axis[i] = (int32_t)(uint32_t)shape[i];
    return axis;
}

result<scalar> stackvm_runtime_function::pop_scalar(datatype_t type) noexcept
{
    try_var(var, stack_.pop());
    scalar s;
    switch (type)
    {
    case dt_int8:
        s = var.as_i1();
        break;
    case dt_int16:
        s = var.as_i2();
        break;
    case dt_int32:
        s = var.as_i4();
        break;
    case dt_uint8:
        s = var.as_u1();
        break;
    case dt_uint16:
        s = var.as_u2();
        break;
    case dt_uint32:
        s = var.as_u4();
        break;
    case dt_bfloat16:
        s = var.as_br2();
        break;
    case dt_float32:
        s = var.as_r4();
        break;
    default:
        return err(std::errc::not_supported);
    }

    return ok(s);
}

result<runtime_tensor> stackvm_runtime_function::create_tensor(uintptr_t addr, datatype_t datatype, const runtime_shape_t &shape, const runtime_shape_t &strides) noexcept
{
    hrt::memory_pool_t pool;
    uintptr_t physical_address = 0;
    auto data_span = module().data();
    auto rdata_span = module().rdata();

    if (addr >= reinterpret_cast<uintptr_t>(data_span.begin())
        && addr < reinterpret_cast<uintptr_t>(data_span.end()))
    {
        auto &tensor = module().data_tensor();
        auto &block = static_cast<const detail::host_runtime_tensor_impl *>(tensor.impl())->memory_block();
        pool = block.pool;
        physical_address = block.physical_block.physical_address + (addr - block.virtual_address);
    }
    else if (addr >= reinterpret_cast<uintptr_t>(rdata_span.begin())
        && addr < reinterpret_cast<uintptr_t>(rdata_span.end()))
    {
        pool = hrt::pool_cpu_only;
    }
    else
    {
        bool found = false;
        for (size_t i = 0; i < inputs_size(); i++)
        {
            try_var(tensor, device_input_tensor(i));
            auto &block = static_cast<detail::host_runtime_tensor_impl &>(*tensor.impl()).memory_block();
            if (addr >= block.virtual_address
                && addr < block.virtual_address + block.size_bytes)
            {
                pool = block.pool;
                physical_address = block.physical_block.physical_address + (addr - block.virtual_address);
                found = true;
                break;
            }
        }

        if (!found)
        {
            for (size_t i = 0; i < outputs_size(); i++)
            {
                try_var(tensor, device_output_tensor(i));
                auto &block = static_cast<detail::host_runtime_tensor_impl &>(*tensor.impl()).memory_block();
                if (addr >= block.virtual_address
                    && addr < block.virtual_address + block.size_bytes)
                {
                    pool = block.pool;
                    physical_address = block.physical_block.physical_address + (addr - block.virtual_address);
                    found = true;
                    break;
                }
            }
        }

        CHECK_WITH_ERR(found, std::errc::invalid_argument);
    }

    auto size = runtime::get_bytes(datatype, shape, strides);
    return hrt::create(datatype, shape, strides, { reinterpret_cast<gsl::byte *>(addr), size }, false, pool, physical_address);
}
