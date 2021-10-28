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
#include "../runtime_function.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const ldc_i4_op_t &op) noexcept
{
    return stack_.push(op.imm);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldnull_op_t &op) noexcept
{
    return stack_.push((uintptr_t)0);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldc_i4_0_op_t &op) noexcept
{
    return stack_.push((int32_t)0);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldc_i4_1_op_t &op) noexcept
{
    return stack_.push((int32_t)1);
}

result<void> stackvm_runtime_function::visit(const ldc_r4_op_t &op) noexcept
{
    return stack_.push(op.imm);
}

#define LDINDIMPL(type)                     \
    try_var(addr, stack_.pop());            \
    if (!addr.as_u())                       \
        return err(std::errc::bad_address); \
    return stack_.push(*reinterpret_cast<const type *>(addr.as_u()))

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_i1_op_t &op) noexcept
{
    LDINDIMPL(int8_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_i2_op_t &op) noexcept
{
    LDINDIMPL(int16_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_i4_op_t &op) noexcept
{
    LDINDIMPL(int32_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_i_op_t &op) noexcept
{
    LDINDIMPL(intptr_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_u1_op_t &op) noexcept
{
    LDINDIMPL(uint8_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_u2_op_t &op) noexcept
{
    LDINDIMPL(uint16_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_u4_op_t &op) noexcept
{
    LDINDIMPL(uint32_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_u_op_t &op) noexcept
{
    LDINDIMPL(uintptr_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_br2_op_t &op) noexcept
{
    LDINDIMPL(bfloat16);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_r4_op_t &op) noexcept
{
    LDINDIMPL(float);
}

#define STINDIMPL(type)                                                                \
    try_var(value, stack_.pop());                                                      \
    try_var(addr, stack_.pop());                                                       \
    if (!addr.as_u())                                                                  \
        return err(std::errc::bad_address);                                            \
    *reinterpret_cast<decltype(value.as_##type()) *>(addr.as_u()) = value.as_##type(); \
    return ok()

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_i1_op_t &op) noexcept
{
    STINDIMPL(i1);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_i2_op_t &op) noexcept
{
    STINDIMPL(i2);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_i4_op_t &op) noexcept
{
    STINDIMPL(i4);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_i_op_t &op) noexcept
{
    STINDIMPL(i);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_br2_op_t &op) noexcept
{
    STINDIMPL(br2);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stind_r4_op_t &op) noexcept
{
    STINDIMPL(r4);
}

result<void> stackvm_runtime_function::visit(const lea_gp_op_t &op) noexcept
{
    try_var(reg, module().reg(op.gpid));
    return stack_.push((intptr_t)reg + op.offset);
}

result<void> stackvm_runtime_function::visit(const lea_buffer_op_t &op) noexcept
{
#define ID_NOT_FOUND ((size_t)-1)
    // TODO: use subres
    if (op.location == mem_input)
    {
        size_t id = ID_NOT_FOUND;
        uint32_t last_start = 0;
        uint32_t offset = 0;
        for (size_t i = 0; i < inputs_size(); i++)
        {
            auto start = input_desc(i).start;
            if (start <= op.offset
                && start >= last_start)
            {
                id = i;
                last_start = start;
                offset = op.offset - start;
            }
        }

        if (id != ID_NOT_FOUND)
        {
            try_var(tensor, device_input_tensor(id));
            try_var(tensor_map, hrt::map(tensor, hrt::map_read));
            return stack_.push((uintptr_t)tensor_map.buffer().data() + offset);
        }
        else
        {
            return err(std::errc::invalid_argument);
        }
    }
    else if (op.location == mem_output)
    {
        size_t id = ID_NOT_FOUND;
        uint32_t last_start = 0;
        uint32_t offset = 0;
        for (size_t i = 0; i < outputs_size(); i++)
        {
            auto start = output_desc(i).start;
            if (start <= op.offset
                && start >= last_start)
            {
                id = i;
                last_start = start;
                offset = op.offset - start;
            }
        }

        if (id != ID_NOT_FOUND)
        {
            try_var(tensor, device_output_tensor(id));
            try_var(tensor_map, hrt::map(tensor, hrt::map_read_write));
            return stack_.push((uintptr_t)tensor_map.buffer().data() + offset);
        }
        else
        {
            return err(std::errc::invalid_argument);
        }
    }
    else if (op.location == mem_rdata)
    {
        auto buffer = module().rdata().subspan(op.offset);
        return stack_.push((uintptr_t)buffer.data());
    }
    else if (op.location == mem_data)
    {
        auto buffer = module().data().subspan(op.offset);
        return stack_.push((uintptr_t)buffer.data());
    }
    else
    {
        return err(std::errc::invalid_argument);
    }
}

#define LDELEM_IMPL(type)          \
    try_var(offset, stack_.pop()); \
    try_var(addr, stack_.pop());   \
    return stack_.push(reinterpret_cast<const type *>(addr.as_u())[offset.as_u()])

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_i1_op_t &op) noexcept
{
    LDELEM_IMPL(int8_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_i2_op_t &op) noexcept
{
    LDELEM_IMPL(int16_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_i4_op_t &op) noexcept
{
    LDELEM_IMPL(int32_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_i_op_t &op) noexcept
{
    LDELEM_IMPL(intptr_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_u1_op_t &op) noexcept
{
    LDELEM_IMPL(uint8_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_u2_op_t &op) noexcept
{
    LDELEM_IMPL(uint16_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_u4_op_t &op) noexcept
{
    LDELEM_IMPL(uint32_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_u_op_t &op) noexcept
{
    LDELEM_IMPL(uintptr_t);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_br2_op_t &op) noexcept
{
    LDELEM_IMPL(bfloat16);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldelem_r4_op_t &op) noexcept
{
    LDELEM_IMPL(float);
}

#define STELEM_IMPL(type)                                                                            \
    try_var(value, stack_.pop());                                                                    \
    try_var(offset, stack_.pop());                                                                   \
    try_var(addr, stack_.pop());                                                                     \
    reinterpret_cast<decltype(value.as_##type()) *>(addr.as_u())[offset.as_u()] = value.as_##type(); \
    return ok()

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_i1_op_t &op) noexcept
{
    STELEM_IMPL(i1);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_i2_op_t &op) noexcept
{
    STELEM_IMPL(i2);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_i4_op_t &op) noexcept
{
    STELEM_IMPL(i4);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_i_op_t &op) noexcept
{
    STELEM_IMPL(i);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_br2_op_t &op) noexcept
{
    STELEM_IMPL(br2);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const stelem_r4_op_t &op) noexcept
{
    STELEM_IMPL(r4);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_0_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_1_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_2_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_3_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_4_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_5_op_t &op) noexcept
{
    return err(std::errc::not_supported);
}

result<void> stackvm_runtime_function::visit(const stshape_op_t &op) noexcept
{
    runtime_shape_t shape;
    try
    {
        shape.resize(op.rank);
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    for (size_t i = 0; i < shape.size(); i++)
    {
        try_var(dim, stack_.pop());
        shape[op.rank - i - 1] = (size_t)dim.as_u();
    }

    return module().shape_reg(op.rshape, std::move(shape));
}

result<void> stackvm_runtime_function::visit(const stpaddings_op_t &op) noexcept
{
    runtime_paddings_t paddings;
    try
    {
        paddings.resize(op.rank);
    }
    catch (...)
    {
        return err(std::errc::not_enough_memory);
    }

    for (size_t i = 0; i < paddings.size(); i++)
    {
        try_var(interior, stack_.pop());
        try_var(after, stack_.pop());
        try_var(before, stack_.pop());
        paddings[op.rank - i - 1] = { before.as_i4(), after.as_i4(), interior.as_i4() };
    }

    return module().paddings_reg(op.rpaddings, std::move(paddings));
}
