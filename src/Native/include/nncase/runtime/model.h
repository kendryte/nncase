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

inline constexpr size_t MAX_SECTION_NAME_LENGTH = 16;
inline constexpr size_t MAX_MODULE_KIND_LENGTH = 16;
inline constexpr uint32_t MODEL_HAS_NO_ENTRY = -1;

typedef std::array<char, MAX_MODULE_KIND_LENGTH> module_kind_t;

template <std::size_t N, std::size_t... Is>
constexpr module_kind_t to_module_kind(const char (&a)[N],
                                       std::index_sequence<Is...>) {
    return {{a[Is]...}};
}

template <std::size_t N>
constexpr module_kind_t to_module_kind(const char (&a)[N]) {
    return to_module_kind(a, std::make_index_sequence<N>());
}

struct model_header {
    uint32_t identifier;
    uint32_t version;
    uint32_t flags;
    uint32_t alignment;
    uint32_t modules;
    uint32_t entry_module;
    uint32_t entry_function;
    uint32_t reserved0;
};

struct function_header {
    uint32_t parameters;
    uint32_t sections;
    uint64_t entrypoint;
    uint64_t text_size;
    uint64_t size;
};

struct module_header {
    module_kind_t kind;
    uint32_t version;
    uint32_t sections;
    uint32_t functions;
    uint32_t reserved0;
    uint64_t size;
};

struct section_header {
    char name[MAX_SECTION_NAME_LENGTH];
    uint32_t flags;
    uint32_t reserved0;
    uint64_t size;
    uint64_t body_start;
    uint64_t body_size;
    uint64_t memory_size;
};

NNCASE_INLINE_VAR constexpr uint32_t SECTION_MERGED_INTO_RDATA = 1;

struct shape_header {
    uint32_t size;

    shape_header() = delete;
    shape_header(shape_header &) = delete;
    shape_header &operator=(shape_header &) = delete;

    const uint32_t *begin() const noexcept {
        return reinterpret_cast<const uint32_t *>(
            reinterpret_cast<uintptr_t>(this) + sizeof(shape_header));
    }

    const uint32_t *end() const noexcept { return begin() + size; }

    uint32_t operator[](size_t index) const {
        assert(index < size);
        return begin()[index];
    }
};

NNCASE_INLINE_VAR constexpr uint32_t MODEL_IDENTIFIER = 'KMDL';
NNCASE_INLINE_VAR constexpr uint32_t MODEL_VERSION = 7;

END_NS_NNCASE_RUNTIME
