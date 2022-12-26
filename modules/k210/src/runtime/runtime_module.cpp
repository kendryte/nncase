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
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/k210/error.h>
#include <nncase/runtime/k210/runtime_types.h>
#include <nncase/runtime/runtime_loader.h>
#ifndef NNCASE_SIMULATOR
#include <kpu.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;
using namespace nncase::runtime::k210;

result<void> k210_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
#ifndef NNCASE_SIMULATOR
    kpu->interrupt_clear.reg = 7;
    kpu->interrupt_mask.reg = 7;
    kpu->fifo_threshold.reg = 10 | (1 << 4);
    kpu->eight_bit_mode.reg = 1;

    plic_set_priority(IRQN_AI_INTERRUPT, 1);
#endif

    assert(context.is_section_pinned());
    auto data_pool = mempool(mem_data);
    if (data_pool.size) {
        data_.reset(new (std::nothrow) gsl::byte[data_pool.size]);
        if (!data_)
            return err(std::errc::not_enough_memory);
    }

    rdata_ = context.section(".rdata");
    text_ = context.section(".text");

#ifndef NNCASE_SIMULATOR
    memcpy((uint8_t *)rdata_.data() - IOMEM, (uint8_t *)rdata_.data(),
           rdata_.size_bytes());
#endif
    return ok();
}

gsl::span<gsl::byte> k210_runtime_module::data() const noexcept {
    return {data_.get(), mempool(mem_data).size};
}

gsl::span<gsl::byte> k210_runtime_module::kpu_ram() noexcept {
    gsl::byte *base;
#ifdef NNCASE_SIMULATOR
    base = kpu_ram_.data();
#else
    base = reinterpret_cast<gsl::byte *>(AI_IO_BASE_ADDR);
#endif
    return {base, KPU_RAM_SIZE};
}

gsl::span<const gsl::byte> k210_runtime_module::rdata() const noexcept {
    return rdata_;
}

result<std::unique_ptr<runtime_function>>
k210_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              k210_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>> k210::create_k210_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            k210_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

extern "C" {
NNCASE_MODULES_K210_API void
RUNTIME_MODULE_ACTIVATOR_NAME(result<std::unique_ptr<runtime_module>> &result) {
    result = create_k210_runtime_module();
}
}

#ifndef NNCASE_SIMULATOR
runtime_registration nncase::runtime::builtin_runtimes[] = {
    {k210_module_type, RUNTIME_MODULE_ACTIVATOR_NAME}, {}};
#endif
