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
#include <nncase/ntt/cpu_runtime.h>
#include <nncase/runtime/runtime_function.h>
#include <nncase/tensor.h>

#if WIN32
#include "loaders/pe/pe_loader.h"
#elif defined(__APPLE__)
#include "loaders/macho/macho_loader.h"
#else
#include "loaders/elf/elf_loader.h"
#endif

BEGIN_NS_NNCASE_RT_MODULE(cpu)

#define CPU_ENTRY_NAME "kernel_entry"

class cpu_runtime_function final : public runtime_function {
    typedef void (*kernel_entry_t)(nncase_runtime_cpu_mt_t *cpu_mt,
                                   gsl::byte **inputs, const gsl::byte *rdata,
                                   gsl::byte *data);

  public:
    cpu_runtime_function(runtime_module &rt_module);
    virtual ~cpu_runtime_function();

    cpu_runtime_module &module() const noexcept;

  protected:
    result<void>
    initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(gsl::span<value_t> parameters,
                                value_t return_value) noexcept override;

  private:
    result<void> run(gsl::span<gsl::byte *> params) noexcept;

  private:
#if WIN32
    pe_loader loader_;
#elif defined(__APPLE__)
    macho_loader loader_;
#else
    elf_loader loader_;
#endif

    kernel_entry_t kernel_entry_;
    uint64_t data_pool_size_;
    uint64_t data_align_;
};

END_NS_NNCASE_RT_MODULE
