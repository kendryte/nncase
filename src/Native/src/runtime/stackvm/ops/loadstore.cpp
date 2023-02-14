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
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

#define MAX_SIGNATURE_SIZE 65535

result<void> stackvm_runtime_function::visit(const ldc_i4_op_t &op) noexcept {
    return stack_.push(op.imm);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldnull_op_t &op) noexcept {
    return stack_.push((uintptr_t)0);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldc_i4_0_op_t &op) noexcept {
    return stack_.push((int32_t)0);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldc_i4_1_op_t &op) noexcept {
    return stack_.push((int32_t)1);
}

result<void> stackvm_runtime_function::visit(const ldc_r4_op_t &op) noexcept {
    return stack_.push(op.imm);
}

#define LDINDIMPL(type)                                                        \
    try_var(addr, stack_.pop());                                               \
    if (!addr.as_u())                                                          \
        return err(std::errc::bad_address);                                    \
    return stack_.push(*reinterpret_cast<const type *>(addr.as_u()))

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_i1_op_t &op) noexcept {
    LDINDIMPL(int8_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_i2_op_t &op) noexcept {
    LDINDIMPL(int16_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_i4_op_t &op) noexcept {
    LDINDIMPL(int32_t);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_i_op_t &op) noexcept {
    LDINDIMPL(intptr_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_u1_op_t &op) noexcept {
    LDINDIMPL(uint8_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_u2_op_t &op) noexcept {
    LDINDIMPL(uint16_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_u4_op_t &op) noexcept {
    LDINDIMPL(uint32_t);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldind_u_op_t &op) noexcept {
    LDINDIMPL(uintptr_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_br2_op_t &op) noexcept {
    LDINDIMPL(bfloat16);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldind_r4_op_t &op) noexcept {
    LDINDIMPL(float);
}

#define STINDIMPL(type)                                                        \
    try_var(value, stack_.pop());                                              \
    try_var(addr, stack_.pop());                                               \
    if (!addr.as_u())                                                          \
        return err(std::errc::bad_address);                                    \
    *reinterpret_cast<decltype(value.as_##type()) *>(addr.as_u()) =            \
        value.as_##type();                                                     \
    return ok()

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stind_i1_op_t &op) noexcept {
    STINDIMPL(i1);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stind_i2_op_t &op) noexcept {
    STINDIMPL(i2);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stind_i4_op_t &op) noexcept {
    STINDIMPL(i4);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const stind_i_op_t &op) noexcept {
    STINDIMPL(i);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stind_br2_op_t &op) noexcept {
    STINDIMPL(br2);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stind_r4_op_t &op) noexcept {
    STINDIMPL(r4);
}

result<void> stackvm_runtime_function::visit(const lea_gp_op_t &op) noexcept {
    try_var(reg, module().reg(op.gpid));
    return stack_.push((intptr_t)reg + op.offset);
}

#define LDELEM_IMPL(type)                                                      \
    try_var(offset, stack_.pop());                                             \
    try_var(addr, stack_.pop());                                               \
    return stack_.push(                                                        \
        reinterpret_cast<const type *>(addr.as_u())[offset.as_u()])

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_i1_op_t &op) noexcept {
    LDELEM_IMPL(int8_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_i2_op_t &op) noexcept {
    LDELEM_IMPL(int16_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_i4_op_t &op) noexcept {
    LDELEM_IMPL(int32_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_i_op_t &op) noexcept {
    LDELEM_IMPL(intptr_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_u1_op_t &op) noexcept {
    LDELEM_IMPL(uint8_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_u2_op_t &op) noexcept {
    LDELEM_IMPL(uint16_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_u4_op_t &op) noexcept {
    LDELEM_IMPL(uint32_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_u_op_t &op) noexcept {
    LDELEM_IMPL(uintptr_t);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_br2_op_t &op) noexcept {
    LDELEM_IMPL(bfloat16);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldelem_r4_op_t &op) noexcept {
    LDELEM_IMPL(float);
}

#define STELEM_IMPL(type)                                                      \
    try_var(value, stack_.pop());                                              \
    try_var(offset, stack_.pop());                                             \
    try_var(addr, stack_.pop());                                               \
    reinterpret_cast<decltype(value.as_##type()) *>(                           \
        addr.as_u())[offset.as_u()] = value.as_##type();                       \
    return ok()

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_i1_op_t &op) noexcept {
    STELEM_IMPL(i1);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_i2_op_t &op) noexcept {
    STELEM_IMPL(i2);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_i4_op_t &op) noexcept {
    STELEM_IMPL(i4);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_i_op_t &op) noexcept {
    STELEM_IMPL(i);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_br2_op_t &op) noexcept {
    STELEM_IMPL(br2);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const stelem_r4_op_t &op) noexcept {
    STELEM_IMPL(r4);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_op_t &op) noexcept {
    try_var(frame, frames_.top());
    try_var(arg, frame->arg(op.index));
    return stack_.push(std::move(arg));
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_0_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 0};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_1_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 1};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_2_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 2};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_3_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 3};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_4_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 4};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldarg_5_op_t &op) noexcept {
    ldarg_op_t ldarg_op{{opcode_t::LDARG}, 5};
    return visit(ldarg_op);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldshape_op_t &op) noexcept {
    dims_t shape;
    try {
        try_var(rank, stack_.pop());
        shape.resize(rank.as_u());
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    for (size_t i = 0; i < shape.size(); i++) {
        try_var(dim, stack_.pop());
        shape[shape.size() - i - 1] = (size_t)dim.as_u();
    }

    return stack_.push(std::move(shape));
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldstrides_op_t &op) noexcept {
    ldshape_op_t new_op;
    return visit(new_op);
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldtuple_elem_op_t &op) noexcept {
    try_var(index, stack_.pop());
    try_var(value, stack_.pop());
    try_var(t, value.as_object().as<tuple>());
    return stack_.push(t->fields()[index.as_u()]);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ldtuple_op_t &op) noexcept {
    try_var(count, stack_.pop());
    std::vector<value_t> fields(count.as_u());
    for (size_t i = 0; i < fields.size(); i++) {
        try_var(field, stack_.pop());
        try_set(fields[i], field.as_object().as<value_t>());
    }
    return stack_.push(tuple(std::in_place, fields));
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const lddatatype_op_t &op) noexcept {
    try_var(addr, stack_.pop());
    span_reader sr(
        {reinterpret_cast<const gsl::byte *>(addr.as_u()), MAX_SIGNATURE_SIZE});
    try_var(dtype, deserialize_datatype(sr));
    return stack_.push(std::move(dtype));
}

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldtensor_op_t &op) noexcept {
    try_var(dtype, pop_object<datatype_t>());
    try_var(shape, pop_shape());
    try_var(strides, pop_shape());
    try_var(addr, stack_.pop());

    gsl::span<gsl::byte> data(reinterpret_cast<gsl::byte *>(addr.as_u()),
                              get_bytes(dtype, shape, strides));
    buffer_attach_options options{};
    try_var(buffer, buffer_allocator::host().attach(data, options));
    return stack_.push(tensor(std::in_place, dtype, shape, strides, buffer));
}

#define RETURN_RESULT_SELECT(RETURN_RESULT_IMPL)                               \
    RETURN_RESULT_IMPL(bool);                                                  \
    RETURN_RESULT_IMPL(int8_t);                                                \
    RETURN_RESULT_IMPL(uint8_t);                                               \
    RETURN_RESULT_IMPL(int32_t);                                               \
    RETURN_RESULT_IMPL(uint32_t);                                              \
    RETURN_RESULT_IMPL(float);

result<void> stackvm_runtime_function::visit(
    NNCASE_UNUSED const ldscalar_op_t &op) noexcept {
    try_var(tensor, pop_tensor());
    try_var(tensor_host, tensor->to_host());
    try_var(tensor_buffer, tensor_host->buffer().as_host());
    try_var(input_map, tensor_buffer.map(map_read));
    auto input = input_map.buffer().data();
#define RETURN_RESULT(_in_type)                                                \
    if (tensor->dtype()->typecode() == op.datatype) {                          \
        _in_type scalar = *reinterpret_cast<const _in_type *>(input);          \
        return stack_.push(stack_entry(scalar));                               \
    }

    RETURN_RESULT_SELECT(RETURN_RESULT);
    return err(nncase_errc::datatype_mismatch);
#undef RETURN_RESULT
}

result<void> stackvm_runtime_function::visit(const ldlocal_op_t &op) noexcept {
    try_var(frame, frames_.top());
    try_var(field, frame->field(op.index));
    return stack_.push(field.as_object());
}

result<void> stackvm_runtime_function::visit(const stlocal_op_t &op) noexcept {
    try_var(frame, frames_.top());
    try_var(field, stack_.pop());
    return frame->field(op.index, std::move(field));
}
