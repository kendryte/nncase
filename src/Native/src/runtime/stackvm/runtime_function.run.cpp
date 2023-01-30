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
#include "ops/control.inl"
#include "ops/conversion.inl"
#include "ops/loadstore.inl"
#include "ops/scalar.inl"
#include "ops/stack.inl"
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/stackvm/op_profile.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void>
stackvm_runtime_function::run(gsl::span<const gsl::byte> text) noexcept {
    reader_ = {text};

    while (!reader_.empty()) {
        auto opcode = reader_.read<opcode_t>();
        if (opcode != opcode_t::TENSOR) {
#ifdef ENABLE_OP_PROFILE
            op_profile p(to_string(opcode));
#endif
            NNCASE_STACKVM_DISPATCH_OP(opcode, reader_)
        } else {
            auto tensor_func = reader_.read_unaligned<tensor_function_t>();
#ifdef ENABLE_OP_PROFILE
            op_profile p(to_string(tensor_func));
#endif
            try_(visit(tensor_func, reader_))
        }
    }

#ifdef ENABLE_OP_PROFILE
    op_profile::print();
#endif
    return ok();
}

uintptr_t stackvm_runtime_function::pc() const noexcept {
    return (uintptr_t)(text_.size_bytes() - reader_.avail());
}

result<void> stackvm_runtime_function::pc(uintptr_t value) noexcept {
    if (value >= text_.size_bytes())
        return err(nncase_errc::stackvm_illegal_target);
    reader_ = span_reader(text_.subspan(value));
    return ok();
}

result<void> stackvm_runtime_function::pc_relative(intptr_t offset) noexcept {
    return pc((uintptr_t)((intptr_t)pc() + offset));
}

uintptr_t stackvm_runtime_function::pop_addr() noexcept {
    return stack_.pop().as_u();
}

dims_t stackvm_runtime_function::pop_shape() noexcept {
    auto len = stack_.pop().as_u();
    dims_t dims(len);
    for (auto &d : dims)
        d = (size_t)stack_.pop().as_u();
    return dims;
}

result<scalar> stackvm_runtime_function::pop_scalar(typecode_t type) noexcept {
    auto var = stack_.pop();
    scalar s;
    switch (type) {
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
