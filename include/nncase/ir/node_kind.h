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
#include <compare>
#include <nncase/runtime/datatypes.h>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace nncase::ir {
struct node_kind {
    uint32_t id;
    std::string_view name;
};

constexpr inline bool operator==(const node_kind &lhs,
                                 const node_kind &rhs) noexcept {
    return lhs.id == rhs.id;
}

#define DEFINE_NEUTRAL_NODEKIND(id, name, value)                               \
    NNCASE_INLINE_VAR constexpr node_kind node_##id{value, #name};
#define DEFINE_NODEKIND(target, id, name, value)                               \
    NNCASE_INLINE_VAR constexpr node_kind node_##target##_##id{value, #name};

#include "node_kind.def"

#undef DEFINE_NEUTRAL_NODEKIND
#undef DEFINE_NODEKIND
} // namespace nncase::ir

namespace std {
template <> struct hash<nncase::ir::node_kind> {
    [[nodiscard]] size_t
    operator()(const nncase::ir::node_kind &opcode) const noexcept {
        return std::hash<uint32_t>()(opcode.id);
    }
};
} // namespace std
