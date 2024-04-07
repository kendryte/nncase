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

k210_runtime_module &k210_runtime_function::module() const noexcept {
    return static_cast<k210_runtime_module &>(runtime_function::module());
}

result<void> k210_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    text_ = context.module_init_context().section(".text").subspan(
        context.header().entrypoint, context.header().text_size);
    return ok();
}

result<runtime_tensor>
k210_runtime_function::allocate_input_tensor(size_t index) noexcept {
    return hrt::create(input_desc(index).datatype, input_shape(index),
                       hrt::pool_shared);
}

result<runtime_tensor>
k210_runtime_function::allocate_output_tensor(size_t index) noexcept {
    return hrt::create(output_desc(index).datatype, output_shape(index),
                       hrt::pool_shared);
}

result<void>
k210_runtime_function::validate_input_tensor(NNCASE_UNUSED size_t index,
                                             runtime_tensor tensor) noexcept {
    if (tensor.is_host() &&
        hrt::memory_pool(tensor).unwrap() == hrt::pool_shared)
        return ok();
    return err(std::errc::invalid_argument);
}

result<void>
k210_runtime_function::validate_output_tensor(NNCASE_UNUSED size_t index,
                                              runtime_tensor tensor) noexcept {
    if (tensor.is_host())
        return ok();
    return err(std::errc::invalid_argument);
}

result<void> k210_runtime_function::invoke_core() noexcept {
    for (size_t i = 0; i < inputs_size(); i++) {
        try_var(input, device_input_tensor(i));
        try_(hrt::sync(input, hrt::sync_write_back));
    }

    try_(visit(text_));
    return ok();
}

result<std::span<std::byte>>
k210_runtime_function::memory_at(const memory_range &mrange) noexcept {
#define ID_NOT_FOUND ((size_t)-1)
    std::byte *base;
    switch (mrange.memory_location) {
    case mem_input: {
        size_t id = ID_NOT_FOUND;
        for (size_t i = 0; i < inputs_size(); i++) {
            if (mrange.start == input_desc(i).start) {
                id = i;
                break;
            }
        }

        if (id != ID_NOT_FOUND) {
            try_var(tensor, device_input_tensor(id));
            base = reinterpret_cast<std::byte *>(
                static_cast<host_runtime_tensor_impl *>(tensor.impl())
                    ->memory_block()
                    .virtual_address -
                mrange.start);
        } else {
            return err(std::errc::invalid_argument);
        }
        break;
    }
    case mem_output: {
        size_t id = ID_NOT_FOUND;
        for (size_t i = 0; i < outputs_size(); i++) {
            if (mrange.start == output_desc(i).start) {
                id = i;
                break;
            }
        }

        if (id != ID_NOT_FOUND) {
            try_var(tensor, device_output_tensor(id));
            try_var(tensor_map, hrt::map(tensor, hrt::map_read_write));
            base = tensor_map.buffer().data() - mrange.start;
        } else {
            return err(std::errc::invalid_argument);
        }
        break;
    }
    case mem_rdata:
        base = const_cast<std::byte *>(module().rdata().data());
        break;
    case mem_data:
        base = module().data().data();
        break;
    case mem_kpu:
        base = module().kpu_ram().data();
        break;
    default:
        return err(nncase_errc::invalid_memory_location);
    }

    return ok(gsl::make_span(base + mrange.start, mrange.size));
}
