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
#include "ids_parser.h"
#include <filesystem>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const extcall_op_t &op) noexcept {
    auto module_id = stack_.pop().as_u();
    auto func_id = stack_.pop().as_u();
    try_var(mod, module().interp().find_module_by_id(module_id));
    try_var(func, mod->find_function_by_id(func_id));

#ifdef NNCASE_DUMP_MANAGER
    auto dump_manager = module().interp().dump_manager();
    auto idPath = lookup_path(dump_manager->dump_path());
    // todo: should do search and only once
    auto name = lookup(idPath, module_id, func_id);
    dump_manager->dump_op(name);
#endif

    std::vector<value_t> params(op.args);
    for (size_t i = 0; i < op.args; i++) {
        try_var(arg, pop_object<value_t>());
#ifdef NNCASE_DUMP_MANAGER
        dump_manager->dump_input(arg, "arg_" + std::to_string(i));
#endif
        params[i] = std::move(arg);
    }

    if (op.is_prim_func) {
        std::vector<value_t> outputs;
        for (size_t i = op.args; i < func->parameters_size(); i++) {
            try_var(type, func->parameter_type(i));
            try_var(ttype, type.as<tensor_type>());
            auto &shape = ttype->shape();
            CHECK_WITH_ERR(shape.is_fixed(), std::errc::invalid_argument);
            dims_t dims;
            for (auto &d : shape)
                dims.push_back(d.fixed_value());
            try_var(t, runtime::detail::create(ttype->dtype(), dims));
            outputs.emplace_back(t);
            params.emplace_back(t);
        }

        try_var(retval, func->invoke(params));
        if (outputs.size() == 1)
            stack_.push(outputs[0]);
        else
            stack_.push(tuple(std::in_place, std::move(outputs)));
    } else {
        try_var(retval, func->invoke(params));
        stack_.push(retval);
    }

#ifdef NNCASE_DUMP_MANAGER
    dump_manager->dump_output(stack_.peek().as_object().as<value_t>().unwrap());
#endif
    return ok();
}

result<void>
stackvm_runtime_function::visit(NNCASE_UNUSED const cuscall_op_t &op) noexcept {
    std::vector<value_t> params(op.args);
#ifdef NNCASE_DUMP_MANAGER
    auto dump_manager = module().interp().dump_manager();
    dump_manager->dump_op(op.registered_name);
#endif
    for (size_t i = 0; i < op.args; i++) {
        try_var(arg, pop_object<value_t>());
#ifdef NNCASE_DUMP_MANAGER
        dump_manager->dump_input(arg, "ar1 96 1000 1这个shapeg_" +
                                          std::to_string(i));
#endif
        params[i] = std::move(arg);
    }

    auto table = module().custom_call_table();
    auto it = table.find(op.registered_name);
    if (it == table.end())
        return err(nncase_errc::stackvm_unknow_custom_call);
    try_var(retval,
            it->second(op.fields_span, params, module().kernel_context()));
#ifdef NNCASE_DUMP_MANAGER
    dump_manager->dump_output(retval);
#endif
    stack_.push(retval);
    return ok();
}
