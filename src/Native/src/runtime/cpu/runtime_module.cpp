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
#include "runtime_module.h"
#include "runtime_function.h"
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

result<void> cpu_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_set(text_, context.get_or_read_section(".text", text_storage_, false));
    try_set(rdata_,
            context.get_or_read_section(".rdata", rdata_storage_, false));
    return ok();
}

kernels::kernel_context &cpu_runtime_module::kernel_context() noexcept {
    auto &context = kernels::default_kernel_context();
#ifdef NNCASE_DUMP_MANAGER
    context.dump_manager = interp().dump_manager();
#endif
    return context;
}

result<std::unique_ptr<runtime_function>>
cpu_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              cpu_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>>
cpu::create_cpu_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            cpu_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
