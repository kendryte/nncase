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
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_set(text_, context.get_or_read_section(".text", text_storage_, false));
    try_set(rdata_,
            context.get_or_read_section(".rdata", rdata_storage_, false));

    regs_[0] = (uintptr_t)rdata_.data();

    // register the external custom call.
    try_(context.read_section(
        ".custom_calls", [this](auto reader, size_t) -> result<void> {
            // custom call section layout:
            // 1. used module numbers
            //    - module_kind_t
            //    - module_kind_t
            auto used_module_counts = reader.template read<uint32_t>();
            for (size_t i = 0; i < used_module_counts; i++) {
                auto kind = reader.template read<module_kind_t>();
                try_var(table, runtime_module::collect(kind));
                for (auto &&p : table) {
                    if (custom_call_table_.find(p.first) !=
                        custom_call_table_.end()) {
                        return err(nncase_errc::stackvm_duplicate_custom_call);
                    }
                    custom_call_table_.insert(p);
                }
            }
            return ok();
        }));
    return ok();
}

result<uintptr_t> stackvm_runtime_module::reg(size_t id) const noexcept {
    CHECK_WITH_ERR(id < regs_.size(), std::errc::result_out_of_range);
    return ok(regs_[id]);
}

result<void> stackvm_runtime_module::reg(size_t id, uintptr_t value) noexcept {
    CHECK_WITH_ERR(id < regs_.size(), std::errc::result_out_of_range);
    regs_[id] = value;
    return ok();
}

const std::unordered_map<std::string, runtime_module::custom_call_type>
stackvm_runtime_module::custom_call_table() const noexcept {
    return custom_call_table_;
}

kernels::kernel_context &stackvm_runtime_module::kernel_context() noexcept {
    auto &context = kernels::default_kernel_context();
#ifdef NNCASE_DUMP_MANAGER
    context.dump_manager = interp().dump_manager();
#endif
    return context;
}

result<std::unique_ptr<runtime_function>>
stackvm_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              stackvm_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>>
stackvm::create_stackvm_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            stackvm_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
stackvm::create_stackvm_custom_calls() {
    std::vector<std::pair<std::string, runtime_module::custom_call_type>>
        calls{};
    return ok(std::move(calls));
}