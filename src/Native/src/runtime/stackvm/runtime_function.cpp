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

stackvm_runtime_function::stackvm_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module), reader_({}) {}

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
    try_(run(text_));

    auto ret = stack_.pop();
    CHECK_WITH_ERR(ret.is_object(), nncase_errc::stackvm_illegal_instruction);
    try_var(ret_val, ret.as_object().as<value_t>());
    if (!return_value.empty()) {
        try_(ret_val->copy_to(return_value));
        return ok(return_value);
    }

    return ok(ret_val);
}
