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
#if defined(NNCASE_XPU_MODULE)
#include "arch/xpu/topology.h"
#else
#include "arch/cpu/topology.h"
#endif

#include "shape.h"
#include <cstddef>
#include <cstdint>
#include <string>

#if defined(NNCASE_CPU_MODULE) || defined(NNCASE_XPU_MODULE)
#include <topology_def.h>
#elif !defined(NNCASE_NTT_TOPOLOGY_DEFINED)
namespace nncase::ntt::distributed {
constexpr std::array<size_t, 1> topology_dims = {1};
using topology_shape_t = ntt::fixed_shape<1, 1, 1>;
}
#endif

namespace nncase::ntt::distributed {
inline constexpr size_t topology_levels =
    static_cast<size_t>(topology::count__);

template <topology Scope>
using program_ids_t = ranked_shape<static_cast<size_t>(Scope) + 1>;

constexpr size_t program_dim(topology topo) noexcept {
    int32_t index =
        static_cast<int32_t>(topo) - (topology_levels - topology_dims.size());
    return index < 0 ? 1 : topology_dims[index];
}

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

bool get_profiler_option() noexcept;

template <topology Scope = (topology)(topology_levels - 1)>
auto program_ids() noexcept {
    auto f = []<size_t... Is>(std::index_sequence<Is...>) {
        return program_ids_t<Scope>{program_id<static_cast<topology>(Is)>()...};
    };
    return f(std::make_index_sequence<static_cast<size_t>(Scope) + 1>());
}

template <topology Scope> class topology_synchronizer;

template <topology Scope = (topology)0> void topology_synchronize() noexcept {
    topology_synchronizer<Scope>::synchronize();
}
} // namespace nncase::ntt::distributed
