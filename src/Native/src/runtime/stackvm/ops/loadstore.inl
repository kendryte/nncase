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

NNCASE_STACKVM_DISPATCH_BEGIN(LDC_I4)
stack_.push(op.imm);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDNULL)
stack_.push(object(nullptr));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDC_I4_0)
stack_.push((int32_t)0);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDC_I4_1)
stack_.push((int32_t)1);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDC_R4)
stack_.push(op.imm);
NNCASE_STACKVM_DISPATCH_END()

#define LDINDIMPL(type)                                                        \
    auto addr = pop_addr();                                                    \
    stack_.push(*reinterpret_cast<const type *>(addr))

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_I1)
LDINDIMPL(int8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_I2)
LDINDIMPL(int16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_I4)
LDINDIMPL(int32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_I)
LDINDIMPL(intptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_U1)
LDINDIMPL(uint8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_U2)
LDINDIMPL(uint16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_U4)
LDINDIMPL(uint32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_U)
LDINDIMPL(uintptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_BR2)
LDINDIMPL(bfloat16);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDIND_R4)
LDINDIMPL(float);
NNCASE_STACKVM_DISPATCH_END()

#define STINDIMPL(type)                                                        \
    auto value = stack_.pop();                                                 \
    auto addr = pop_addr();                                                    \
    *reinterpret_cast<decltype(value.as_##type()) *>(addr) = value.as_##type()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_I1)
STINDIMPL(i1);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_I2)
STINDIMPL(i2);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_I4)
STINDIMPL(i4);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_I)
STINDIMPL(i);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_BR2)
STINDIMPL(br2);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STIND_R4)
STINDIMPL(r4);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LEA_GP)
try_var(reg, module().reg(op.gpid));
stack_.push((intptr_t)reg + op.offset);
NNCASE_STACKVM_DISPATCH_END()

#define LDELEM_IMPL(type)                                                      \
    auto offset = stack_.pop();                                                \
    auto addr = pop_addr();                                                    \
    stack_.push(reinterpret_cast<const type *>(addr)[offset.as_i()])

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_I1)
LDELEM_IMPL(int8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_I2)
LDELEM_IMPL(int16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_I4)
LDELEM_IMPL(int32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_I)
LDELEM_IMPL(intptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_U1)
LDELEM_IMPL(uint8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_U2)
LDELEM_IMPL(uint16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_U4)
LDELEM_IMPL(uint32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_U)
LDELEM_IMPL(uintptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_BR2)
LDELEM_IMPL(bfloat16);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDELEM_R4)
LDELEM_IMPL(float);
NNCASE_STACKVM_DISPATCH_END()

#define STELEM_IMPL(type)                                                      \
    auto value = stack_.pop();                                                 \
    auto offset = stack_.pop();                                                \
    auto addr = pop_addr();                                                    \
    reinterpret_cast<decltype(value.as_##type()) *>(addr)[offset.as_i()] =     \
        value.as_##type()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_I1)
STELEM_IMPL(i1);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_I2)
STELEM_IMPL(i2);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_I4)
STELEM_IMPL(i4);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_I)
STELEM_IMPL(i);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_BR2)
STELEM_IMPL(br2);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STELEM_R4)
STELEM_IMPL(r4);
NNCASE_STACKVM_DISPATCH_END()

#define LDARG_IMPL(index)                                                      \
    try_var(frame, frames_.top());                                             \
    auto arg = frame->arg(index);                                              \
    stack_.push(std::move(arg))

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG)
LDARG_IMPL(op.index);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_0)
LDARG_IMPL(0);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_1)
LDARG_IMPL(1);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_2)
LDARG_IMPL(2);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_3)
LDARG_IMPL(3);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_4)
LDARG_IMPL(4);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDARG_5)
LDARG_IMPL(5);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDTUPLE_ELEM)
auto index = stack_.pop_nonobject<size_t>();
auto value = stack_.pop_object();
try_var(t, value.as<tuple>());
stack_.push(t->fields()[index]);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDTUPLE)
auto count = stack_.pop_nonobject<size_t>();
std::vector<value_t> fields(count);
for (auto &field : fields) {
    auto value = stack_.pop_object();
    try_set(field, value.as<value_t>());
}

stack_.push(tuple(std::in_place, std::move(fields)));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDDATATYPE)
auto addr = pop_addr();
span_reader sr({reinterpret_cast<const std::byte *>(addr), MAX_SIGNATURE_SIZE});
try_var(dtype, deserialize_datatype(sr));
stack_.push(std::move(dtype));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDTENSOR)
try_var(dtype, pop_object<datatype_t>());
auto shape = pop_shape();
auto strides = pop_shape();
auto addr = pop_addr();

std::span<std::byte> data(reinterpret_cast<std::byte *>(addr),
                          get_bytes(dtype, shape, strides));
buffer_attach_options options{};
try_var(buffer, buffer_allocator::host().attach(data, options));
stack_.push(tensor(std::in_place, dtype, shape, strides, buffer));
NNCASE_STACKVM_DISPATCH_END()

#define RETURN_RESULT_SELECT(RETURN_RESULT_IMPL)                               \
    RETURN_RESULT_IMPL(dt_boolean, bool);                                      \
    RETURN_RESULT_IMPL(dt_int8, int8_t);                                       \
    RETURN_RESULT_IMPL(dt_uint8, uint8_t);                                     \
    RETURN_RESULT_IMPL(dt_int32, int32_t);                                     \
    RETURN_RESULT_IMPL(dt_uint32, uint32_t);                                   \
    RETURN_RESULT_IMPL(dt_float32, float);

NNCASE_STACKVM_DISPATCH_BEGIN(LDSCALAR)
try_var(tensor, pop_tensor());
try_var(tensor_host, tensor->to_host());
try_var(tensor_buffer, tensor_host->buffer().as_host());
try_var(input_map, tensor_buffer.map(map_read));
auto input = input_map.buffer().data();

#define RETURN_RESULT(_typecode, _in_type)                                     \
    case _typecode: {                                                          \
        _in_type scalar = *reinterpret_cast<const _in_type *>(input);          \
        stack_.push(stack_entry(scalar));                                      \
        break;                                                                 \
    }

switch (tensor->dtype()->typecode()) {
    RETURN_RESULT_SELECT(RETURN_RESULT)
default:
    return err(nncase_errc::datatype_mismatch);
}
#undef RETURN_RESULT
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(LDLOCAL)
try_var(frame, frames_.top());
auto field = frame->field(op.index);
stack_.push(std::move(field));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(STLOCAL)
try_var(frame, frames_.top());
auto field = stack_.pop();
frame->field(op.index, std::move(field));
NNCASE_STACKVM_DISPATCH_END()
