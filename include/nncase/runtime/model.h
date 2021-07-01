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
#include "datatypes.h"
#include <cassert>

BEGIN_NS_NNCASE_RUNTIME

NNCASE_INLINE_VAR constexpr size_t MAX_SECTION_NAME_LENGTH = 16;

struct model_header
{
    uint32_t identifier;
    uint32_t version;
    uint32_t flags;
    uint32_t alignment;
    uint32_t modules;
    uint32_t main_module;
};

struct module_header
{
    module_type_t type;
    uint32_t size;
    uint32_t mempools;
    uint32_t inputs;
    uint32_t outputs;
    uint32_t sections;
    uint32_t reserved0;
};

struct mempool_desc
{
    memory_location_t location;
    uint32_t size;
};

struct section_header
{
    char name[MAX_SECTION_NAME_LENGTH];
    uint32_t flags;
    uint32_t start;
    uint32_t size;
    uint32_t reserved0;
};

NNCASE_INLINE_VAR constexpr uint32_t SECTION_MERGED_INTO_RDATA = 1;

struct shape_header
{
    uint32_t size;

    shape_header() = delete;
    shape_header(shape_header &) = delete;
    shape_header &operator=(shape_header &) = delete;

    const uint32_t *begin() const noexcept
    {
        return reinterpret_cast<const uint32_t *>(reinterpret_cast<uintptr_t>(this) + sizeof(shape_header));
    }

    const uint32_t *end() const noexcept
    {
        return begin() + size;
    }

    uint32_t operator[](size_t index) const
    {
        assert(index < size);
        return begin()[index];
    }
};

NNCASE_INLINE_VAR constexpr uint32_t MODEL_IDENTIFIER = 'KMDL';
NNCASE_INLINE_VAR constexpr uint32_t MODEL_VERSION = 5;

END_NS_NNCASE_RUNTIME
