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
#include "call_frame.h"
#include "evaluate_stack.h"
#include "runtime_module.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/runtime_function.h>
#include <nncase/runtime/stackvm/op_reader.h>
#include <nncase/tensor.h>

BEGIN_NS_NNCASE_RT_MODULE(stackvm)

class stackvm_runtime_function : public runtime_function, private op_visitor {
  public:
    using runtime_function::runtime_function;

    stackvm_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(gsl::span<value_t> parameters,
                                value_t return_value) noexcept override;

    using op_visitor::visit;
#include "runtime_function_ops.h"

  private:
    uintptr_t pc() const noexcept;
    result<void> pc(uintptr_t value) noexcept;
    result<void> pc_relative(intptr_t offset) noexcept;
    result<uintptr_t> pop_addr() noexcept;
    result<scalar> pop_scalar(typecode_t type) noexcept;
    result<dims_t> pop_shape() noexcept;

    template <class T> result<T> pop_object() noexcept {
        try_var(var, stack_.pop());
        if (var.is_object())
            return var.as_object().as<T>();
        return err(std::errc::invalid_argument);
    }

    result<tensor> pop_tensor() noexcept { return pop_object<tensor>(); }

    result<value_t> pop_value() noexcept { return pop_object<value_t>(); }

    template <class T> result<T> pop_addr() noexcept {
        try_var(addr, pop_addr());
        return reinterpret_cast<T>(addr);
    }

  private:
    gsl::span<const gsl::byte> text_;
    evaluate_stack stack_;
    call_frames frames_;
};

END_NS_NNCASE_RT_MODULE
