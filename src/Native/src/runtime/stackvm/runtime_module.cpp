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
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

gsl::span<const gsl::byte> stackvm_runtime_module::rdata() const noexcept {
    return rdata_;
}

result<void> stackvm_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    assert(context.is_section_pinned());
    rdata_ = context.section(".rdata");
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

kernels::kernel_context &stackvm_runtime_module::kernel_context() noexcept {
    return kernels::default_kernel_context();
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
