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
#pragma once
#include "evaluate_stack.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/stackvm/runtime_module.h>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class stackvm_runtime_module : public runtime_module
{
public:
    static NNCASE_INLINE_VAR constexpr size_t MAX_GENERAL_REGS = 32;

    kernels::kernel_context &kernel_context() noexcept;

    gsl::span<gsl::byte> data() const noexcept;
    gsl::span<const gsl::byte> rdata() const noexcept;

    const runtime_tensor &data_tensor() const noexcept;

    result<uintptr_t> reg(size_t id) const noexcept;
    result<void> reg(size_t id, uintptr_t value) noexcept;

    result<runtime_shape_t> shape_reg(size_t id) const noexcept;
    result<void> shape_reg(size_t id, runtime_shape_t value) noexcept;

    result<runtime_paddings_t> paddings_reg(size_t id) const noexcept;
    result<void> paddings_reg(size_t id, runtime_paddings_t value) noexcept;

protected:
    result<void> initialize_before_functions(runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>> create_function() noexcept override;

private:
    runtime_tensor data_;
    gsl::span<const gsl::byte> rdata_;
    std::array<uintptr_t, MAX_GENERAL_REGS> regs_;
    std::vector<runtime_shape_t> shape_regs_;
    std::vector<runtime_paddings_t> paddings_regs_;
};

END_NS_NNCASE_RT_MODULE
