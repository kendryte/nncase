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
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

stackvm_runtime_module &stackvm_runtime_function::module() const noexcept {
    return static_cast<stackvm_runtime_module &>(runtime_function::module());
}

result<void> stackvm_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    text_ = context.module_init_context().section(".text").subspan(
        context.header().entrypoint, context.header().text_size);
    return ok();
}

result<value_t> stackvm_runtime_function::invoke_core(
    gsl::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {
    try_var(frame, frames_.push(0));
    for (auto arg : parameters) {
        try_(frame->push_back_arg(std::move(arg)));
    }

    //    module().interp().options().get<std::string>("dump_path");
    try_(visit(text_));

    checked_try_var(ret, stack_.pop());
    CHECK_WITH_ERR(ret.is_object(), nncase_errc::stackvm_illegal_instruction);
    try_var(ret_val, ret.as_object().as<value_t>());
    if (!return_value.empty()) {
        try_(ret_val->copy_to(return_value));
        return ok(return_value);
    }

    return ok(ret_val);
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

result<uintptr_t> stackvm_runtime_function::pop_addr() noexcept {
    try_var(addr, stack_.pop());
    return ok(addr.as_u());
}

result<dims_t> stackvm_runtime_function::pop_shape() noexcept {
    try_var(var, stack_.pop());
    if (var.is_shape())
        return ok(var.as_shape());
    return err(std::errc::invalid_argument);
}

result<scalar> stackvm_runtime_function::pop_scalar(typecode_t type) noexcept {
    try_var(var, stack_.pop());
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
