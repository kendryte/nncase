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

#define MAX_SIGNATURE_SIZE 65535

#define NNCASE_STACKVM_DISPATCH_OP_LDC_I4 stack_.push(op.imm);

#define NNCASE_STACKVM_DISPATCH_OP_LDNULL stack_.push(object(nullptr));

#define NNCASE_STACKVM_DISPATCH_OP_LDC_I4_0 stack_.push((int32_t)0);

#define NNCASE_STACKVM_DISPATCH_OP_LDC_I4_1 stack_.push((int32_t)1);

#define NNCASE_STACKVM_DISPATCH_OP_LDC_R4 stack_.push(op.imm);

#define LDINDIMPL(type)                                                        \
    auto addr = pop_addr();                                                    \
    stack_.push(*reinterpret_cast<const type *>(addr))

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_I1 LDINDIMPL(int8_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_I2 LDINDIMPL(int16_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_I4 LDINDIMPL(int32_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_I LDINDIMPL(intptr_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_U1 LDINDIMPL(uint8_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_U2 LDINDIMPL(uint16_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_U4 LDINDIMPL(uint32_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_U LDINDIMPL(uintptr_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_BR2 LDINDIMPL(bfloat16);

#define NNCASE_STACKVM_DISPATCH_OP_LDIND_R4 LDINDIMPL(float);

#define STINDIMPL(type)                                                        \
    auto value = stack_.pop();                                                 \
    auto addr = pop_addr();                                                    \
    *reinterpret_cast<decltype(value.as_##type()) *>(addr) = value.as_##type()

#define NNCASE_STACKVM_DISPATCH_OP_STIND_I1 STINDIMPL(i1);

#define NNCASE_STACKVM_DISPATCH_OP_STIND_I2 STINDIMPL(i2);

#define NNCASE_STACKVM_DISPATCH_OP_STIND_I4 STINDIMPL(i4);

#define NNCASE_STACKVM_DISPATCH_OP_STIND_I STINDIMPL(i);

#define NNCASE_STACKVM_DISPATCH_OP_STIND_BR2 STINDIMPL(br2);

#define NNCASE_STACKVM_DISPATCH_OP_STIND_R4 STINDIMPL(r4);

#define NNCASE_STACKVM_DISPATCH_OP_LEA_GP                                      \
    try_var(reg, module().reg(op.gpid));                                       \
    stack_.push((intptr_t)reg + op.offset);

#define LDELEM_IMPL(type)                                                      \
    auto offset = stack_.pop();                                                \
    auto addr = pop_addr();                                                    \
    stack_.push(reinterpret_cast<const type *>(addr)[offset.as_i()])

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_I1 LDELEM_IMPL(int8_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_I2 LDELEM_IMPL(int16_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_I4 LDELEM_IMPL(int32_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_I LDELEM_IMPL(intptr_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_U1 LDELEM_IMPL(uint8_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_U2 LDELEM_IMPL(uint16_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_U4 LDELEM_IMPL(uint32_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_U LDELEM_IMPL(uintptr_t);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_BR2 LDELEM_IMPL(bfloat16);

#define NNCASE_STACKVM_DISPATCH_OP_LDELEM_R4 LDELEM_IMPL(float);

#define STELEM_IMPL(type)                                                      \
    auto value = stack_.pop();                                                 \
    auto offset = stack_.pop();                                                \
    auto addr = pop_addr();                                                    \
    reinterpret_cast<decltype(value.as_##type()) *>(addr)[offset.as_i()] =     \
        value.as_##type()

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_I1 STELEM_IMPL(i1);

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_I2 STELEM_IMPL(i2);

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_I4 STELEM_IMPL(i4);

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_I STELEM_IMPL(i);

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_BR2 STELEM_IMPL(br2);

#define NNCASE_STACKVM_DISPATCH_OP_STELEM_R4 STELEM_IMPL(r4);

#define LDARG_IMPL(index)                                                      \
    try_var(frame, frames_.top());                                             \
    auto arg = frame->arg(index);                                              \
    stack_.push(std::move(arg))

#define NNCASE_STACKVM_DISPATCH_OP_LDARG LDARG_IMPL(op.index);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_0 LDARG_IMPL(0);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_1 LDARG_IMPL(1);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_2 LDARG_IMPL(2);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_3 LDARG_IMPL(3);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_4 LDARG_IMPL(4);

#define NNCASE_STACKVM_DISPATCH_OP_LDARG_5 LDARG_IMPL(5);

#define NNCASE_STACKVM_DISPATCH_OP_LDTUPLE_ELEM                                \
    auto index = stack_.pop();                                                 \
    auto value = stack_.pop();                                                 \
    try_var(t, value.as_object().as<tuple>());                                 \
    stack_.push(t->fields()[index.as_u()]);

#define NNCASE_STACKVM_DISPATCH_OP_LDTUPLE                                     \
    auto count = stack_.pop().as_u();                                          \
    std::vector<value_t> fields(count);                                        \
    for (auto &field : fields) {                                               \
        auto value = stack_.pop().as_object();                                 \
        try_set(field, value.as<value_t>());                                   \
    }                                                                          \
                                                                               \
    stack_.push(tuple(std::in_place, std::move(fields)));

#define NNCASE_STACKVM_DISPATCH_OP_LDDATATYPE                                  \
    auto addr = pop_addr();                                                    \
    span_reader sr(                                                            \
        {reinterpret_cast<const gsl::byte *>(addr), MAX_SIGNATURE_SIZE});      \
    try_var(dtype, deserialize_datatype(sr));                                  \
    stack_.push(std::move(dtype));

#define NNCASE_STACKVM_DISPATCH_OP_LDTENSOR                                    \
    try_var(dtype, pop_object<datatype_t>());                                  \
    auto shape = pop_shape();                                                  \
    auto strides = pop_shape();                                                \
    auto addr = pop_addr();                                                    \
                                                                               \
    gsl::span<gsl::byte> data(reinterpret_cast<gsl::byte *>(addr),             \
                              get_bytes(dtype, shape, strides));               \
    buffer_attach_options options{};                                           \
    try_var(buffer, buffer_allocator::host().attach(data, options));           \
    stack_.push(tensor(std::in_place, dtype, shape, strides, buffer));

#define NNCASE_STACKVM_DISPATCH_OP_LDLOCAL                                     \
    try_var(frame, frames_.top());                                             \
    auto field = frame->field(op.index);                                       \
    stack_.push(std::move(field));

#define NNCASE_STACKVM_DISPATCH_OP_STLOCAL                                     \
    try_var(frame, frames_.top());                                             \
    auto field = stack_.pop();                                                 \
    frame->field(op.index, std::move(field));
