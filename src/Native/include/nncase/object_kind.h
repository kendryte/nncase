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
#include "compiler_defs.h"
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace nncase {
struct object_kind {
    uint32_t id;
    std::string_view name;
};

constexpr inline bool operator==(const object_kind &lhs,
                                 const object_kind &rhs) noexcept {
    return lhs.id == rhs.id;
}

#define DEFINE_OBJECT_KIND(id, name, value)                                    \
    inline constexpr object_kind object_##id{value, #name};

#include "object_kind.def"

#undef DEFINE_OBJECT_KIND
} // namespace nncase

namespace std {
template <> struct hash<nncase::object_kind> {
    [[nodiscard]] size_t
    operator()(const nncase::object_kind &opcode) const noexcept {
        return std::hash<uint32_t>()(opcode.id);
    }
};
} // namespace std
