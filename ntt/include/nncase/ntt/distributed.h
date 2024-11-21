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
#include "arch/cpu/topology.h"
#include <cstddef>
#include <cstdint>

#ifdef NNCASE_CPU_MODULE
#include <topology_def.h>
#endif

namespace nncase::ntt::distributed {
inline constexpr size_t topology_levels =
    static_cast<size_t>(topology::count__);

#ifndef NNCASE_CPU_MODULE
constexpr size_t program_dim(topology /* topo */) noexcept { return 1; }
#else
constexpr size_t program_dim(topology topo) noexcept {
    int32_t index =
        static_cast<int32_t>(topo) - (topology_levels - topology_dims.size());
    return index < 0 ? 1 : topology_dims[index];
}
#endif

template <topology Scope = static_cast<topology>(topology_levels - 1)>
constexpr size_t topology_size() noexcept {
    return [] {
        size_t size = 1;
        for (size_t i = 0; i <= static_cast<size_t>(Scope); i++) {
            size *= program_dim(static_cast<topology>(i));
        }
        return size;
    }();
}

template <topology Topology> struct program_id_getter {
    static size_t id() noexcept;
};

template <topology Topology> size_t program_id() noexcept {
    return program_id_getter<Topology>::id();
}
} // namespace nncase::ntt::distributed
