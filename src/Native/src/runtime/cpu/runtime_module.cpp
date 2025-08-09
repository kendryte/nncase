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
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <string_view>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

typedef struct {
    uint32_t tdim;
    uint32_t bdim;
    uint32_t cdim;
    uint32_t reserved0;
} module_desc_header;

cpu_runtime_module::cpu_runtime_module() noexcept
    : tdim_(0), bdim_(0), cdim_(0) {
#ifdef __APPLE__
    pthread_key_create(&cpu_thread_context_key_,
                       [](void *ptr) { delete (cpu_thread_context_t *)ptr; });
#endif
}

cpu_runtime_module::~cpu_runtime_module() {
#ifdef __APPLE__
    pthread_key_delete(cpu_thread_context_key_);
#endif
}

result<void> cpu_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_(context.read_section(
        ".desc", [this](auto reader, size_t) -> result<void> {
            auto header = reader.template read<module_desc_header>();
            this->tdim_ = header.tdim;
            this->bdim_ = header.bdim;
            this->cdim_ = header.cdim;
            return ok();
        }));

    try_(initialize_text(context));
    try_set(rdata_,
            context.get_or_read_section(".rdata", rdata_storage_, false));
    try_set(thread_local_rdata_,
            context.get_or_read_section(".thread_local_rdata",
                                        thread_local_rdata_storage_, false));
    try_set(block_local_rdata_,
            context.get_or_read_section(".block_local_rdata",
                                        block_local_rdata_storage_, false));
    return ok();
}
result<void> cpu_runtime_module::initialize_text(
    runtime_module_init_context &context) noexcept {
    auto cpu_external_module_path =
        context.interp().options().get<std::string>("cpu_external_module_path");
    if (cpu_external_module_path.is_ok() &&
        !cpu_external_module_path.unwrap().empty()) {
        loader_.load_from_file(cpu_external_module_path.unwrap());
    } else {
        try_set(text_,
                context.get_or_read_section(".text", text_storage_, false));
        loader_.load(text_);
    }

    return ok();
}

result<block_entry_t> cpu_runtime_module::block_entry() const noexcept {
    return ok((block_entry_t)loader_.entry());
}

result<std::unique_ptr<runtime_function>>
cpu_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              cpu_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>> cpu::create_cpu_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            cpu_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
