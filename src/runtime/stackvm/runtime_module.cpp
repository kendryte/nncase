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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_module::initialize_core(runtime_module_init_context &context) noexcept
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

result<runtime_tensor> stackvm_runtime_module::allocate_input_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(input_desc(index).datatype, input_shape(index));
}

result<runtime_tensor> stackvm_runtime_module::allocate_output_tensor(size_t index) noexcept
{
    return host_runtime_tensor::create(output_desc(index).datatype, output_shape(index));
}

result<void> stackvm_runtime_module::validate_input_tensor([[maybe_unused]] size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> stackvm_runtime_module::validate_output_tensor([[maybe_unused]] size_t index, runtime_tensor tensor) noexcept
{
    if (tensor.is_host())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> stackvm_runtime_module::run_core() noexcept
{
    call_depth_ = 0;
    return visit(text_);
}

uintptr_t stackvm_runtime_module::pc() const noexcept
{
    return (uintptr_t)(text_.size_bytes() - reader_.avail());
}

result<void> stackvm_runtime_module::pc(uintptr_t value) noexcept
{
    if (value >= text_.size_bytes())
        return err(nncase_errc::stackvm_illegal_target);
    reader_ = span_reader(text_.subspan(value));
    return ok();
}

result<void> stackvm_runtime_module::pc_relative(intptr_t offset) noexcept
{
    return pc((uintptr_t)((intptr_t)pc() + offset));
}

result<padding> stackvm_runtime_module::pop_padding() noexcept
{
    try_var(after, stack_.pop());
    try_var(before, stack_.pop());
    return ok(padding { before.as_i4(), after.as_i4() });
}

result<uintptr_t> stackvm_runtime_module::pop_addr() noexcept
{
    try_var(addr, stack_.pop());
    return ok(addr.as_u());
}

runtime_axis_t stackvm_runtime_module::as_runtime_axis(const runtime_shape_t &shape)
{
    runtime_axis_t axis(shape.size());
    for (size_t i = 0; i < shape.size(); i++)
        axis[i] = (int32_t)(uint32_t)shape[i];
    return axis;
}

result<scalar> stackvm_runtime_module::pop_scalar(datatype_t type) noexcept
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

result<std::unique_ptr<runtime_module>> stackvm::create_stackvm_runtime_module()
{
    std::unique_ptr<runtime_module> mod(new (std::nothrow) stackvm_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
