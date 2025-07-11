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
#include "section.h"
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_section_context.h>

using namespace nncase;
using namespace nncase::runtime;

result<std::span<const std::byte>> runtime_section_context::get_or_read_section(
    const char *name, buffer_t &buffer_storage, bool allocate_shared) noexcept {
    return runtime_section_context::get_or_read_section(name, buffer_storage, allocate_shared, nullptr);
}