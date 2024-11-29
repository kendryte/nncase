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
#include "runtime_function.h"
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

typedef struct {
    uint32_t tdim;
    uint32_t bdim;
    uint32_t cdim;
    uint32_t reserved0;
} desc_header;

cpu_runtime_function::cpu_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module), block_entry_(nullptr), tdim_(0), bdim_(0) {}

cpu_runtime_function::~cpu_runtime_function() {}

cpu_runtime_module &cpu_runtime_function::module() const noexcept {
    return static_cast<cpu_runtime_module &>(runtime_function::module());
}

result<void> cpu_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    try_(context.read_section(
        ".desc", [this](auto reader, size_t) -> result<void> {
            auto header = reader.template read<desc_header>();
            this->tdim_ = header.tdim;
            this->bdim_ = header.bdim;
            this->cdim_ = header.cdim;
            return ok();
        }));
    auto text = module().text().subspan(context.header().entrypoint,
                                        context.header().text_size);
    loader_.load(text);
    block_entry_ = (block_entry_t)loader_.entry();
    return ok();
}

result<value_t> cpu_runtime_function::invoke_core(
    std::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {
    std::vector<std::byte *> param_ptrs;
    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        try_var(hb, t->buffer().as_host());
        try_var(m, hb.map(map_read_write));
        param_ptrs.emplace_back(m.buffer().data());
        m.release();
    }

    try_(run(param_ptrs));

    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        try_var(hb, t->buffer().as_host());
        try_(hb.unmap());
    }

    return ok(tuple(std::in_place));
}
