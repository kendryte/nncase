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
#include <nncase/runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const nop_op_t &op) noexcept {
    return ok();
}

result<void> stackvm_runtime_function::visit(const br_op_t &op) noexcept {
    return pc_relative(op.target);
}

result<void> stackvm_runtime_function::visit(const br_true_op_t &op) noexcept {
    try_var(value, stack_.pop());
    if (value.as_i())
        return pc_relative(op.target);
    return ok();
}

result<void> stackvm_runtime_function::visit(const br_false_op_t &op) noexcept {
    try_var(value, stack_.pop());
    if (!value.as_i())
        return pc_relative(op.target);
    return ok();
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ret_op_t &op) noexcept {
    try_var(ret_addr, frames_.pop());
    if (frames_.empty()) {
        interrupted_ = true;
        return ok();
    }

    return pc(ret_addr);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const call_op_t &op) noexcept {
    return err(std::errc::not_supported);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const ecall_op_t &op) noexcept {
    return err(std::errc::not_supported);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const extcall_op_t &op) noexcept {
    try_var(module_id, stack_.pop());
    try_var(func_id, stack_.pop());
    try_var(mod, module().interp().find_module_by_id(module_id.as_u()));
    try_var(func, mod->find_function_by_id(func_id.as_u()));

    std::vector<value_t> params(op.args);
    for (size_t i = 0; i < op.args; i++) {
        try_var(arg, pop_object<value_t>());
        params[i] = std::move(arg);
    }

    try_var(retval, func->invoke(params));
    return stack_.push(retval);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const cuscall_op_t &op) noexcept {
    std::vector<value_t> params(op.args);
    for (size_t i = 0; i < op.args; i++) {
        try_var(arg, pop_object<value_t>());
        params[i] = std::move(arg);
    }
    auto table = module().custom_call_table();
    auto it = table.find(op.registered_name);
    if (it == table.end())
        return err(nncase_errc::stackvm_unknow_custom_call);
    try_var(retval,
            it->second(op.fields_span, params, module().kernel_context()));
    return stack_.push(retval);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const throw_op_t &op) noexcept {
    return err(std::errc::not_supported);
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const break_op_t &op) noexcept {
    return err(std::errc::not_supported);
}
