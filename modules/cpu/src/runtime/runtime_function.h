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
#include "runtime_module.h"
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/runtime_function.h>

BEGIN_NS_NNCASE_RT_MODULE(cpu)

class cpu_runtime_function : public runtime_function {
  public:
    using runtime_function::runtime_function;

    cpu_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(gsl::span<value_t> parameters,
                                value_t return_value) noexcept override;

  private:
    gsl::span<const gsl::byte> text_;
    // std::vector<uint32_t> input_ranks_;
    // std::vector<std::vector<uint32_t>> input_shapes_;
    // std::vector<std::vector<uint32_t>> input_strides_;
    // std::vector<uint32_t> output_ranks_;
    // std::vector<std::vector<uint32_t>> output_shapes_;
    // std::vector<std::vector<uint32_t>> output_strides_;
};

END_NS_NNCASE_RT_MODULE
