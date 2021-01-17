/* Copyright 2019-2020 Canaan Inc.
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
#include "runtime_loader.h"
#include <cassert>
#include <iostream>
#include <nncase/runtime/error.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/span_reader.h>

using namespace nncase;
using namespace nncase::runtime;

interpreter::interpreter() noexcept
{
}

result<void> interpreter::load_model(gsl::span<const gsl::byte> buffer) noexcept
{
    span_reader reader(buffer);
    auto header = reader.get_ref<model_header>();
    // 1. Validate model
    if (header->identifier != MODEL_IDENTIFIER)
        return make_error_condition(nncase_errc::invalid_model_indentifier);
    if (header->version != MODEL_VERSION)
        return make_error_condition(nncase_errc::invalid_model_version);
    // TODO: Validate checksum

    // 2. Load modules
    entry_module_ = header->entry_module;
    span_reader content(reader.peek_avail().subspan(sizeof(module_header) * header->modules));
    for (size_t i = 0; i < header->modules; i++)
    {
        auto header = reader.get_ref<module_header>();
        auto module_body = content.read_span(header->size);
        try_var(rt_module, runtime_module::create(header->type));
        modules_.emplace_back(std::move(rt_module));
    }

    return ok();
}
