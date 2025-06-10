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
#include "../arch/xpu/topology_def.h"
#else
#include "../arch/cpu/topology_def.h"
#endif

#include "../shape.h"
#include <cstddef>

#if defined(NNCASE_CPU_MODULE) || defined(NNCASE_XPU_MODULE)
#include <module_topology_def.h>
#elif !defined(NNCASE_NTT_TOPOLOGY_DEFINED)
namespace nncase::ntt::distributed {
constexpr auto topology_shape = ntt::fixed_shape_v<1, 1, 1>;
} // namespace nncase::ntt::distributed
#endif

namespace nncase::ntt::distributed {
inline constexpr size_t topology_levels =
    static_cast<size_t>(topology::count__);

template <class T, topology Scope>
concept ScopedProgramIds =
    Shape<T> && T::rank() == static_cast<size_t>(Scope) + 1;

template <topology Scope>
using dynamic_program_ids_t = dynamic_shape_t<static_cast<size_t>(Scope) + 1>;

/**
 * @brief Get the dimension of the program for a given topology.
 * @tparam Topology The topology level for which to get the dimension.
 */
template <topology Topology> constexpr auto program_dim() noexcept {
    constexpr dim_t index = static_cast<dim_t>(Topology) -
                            (topology_levels - topology_shape.rank());
    if constexpr (index < 0) {
        return dim_one;
    } else {
        return topology_shape[fixed_dim_v<index>];
    }
}

/**
 * @brief Get the total size of the topology from top to the given topology.
 * @tparam Scope The topology scope for which to get the size.
 */
template <topology Scope = static_cast<topology>(topology_levels - 1)>
constexpr auto topology_up_size() noexcept {
    auto impl = []<size_t... Is>(std::index_sequence<Is...>) {
        return (... * program_dim<static_cast<topology>(Is)>());
    };
    return impl(std::make_index_sequence<static_cast<size_t>(Scope) + 1>());
}

template <topology Topology> struct program_id_getter {
    static dim_t id() noexcept;
};

template <topology Topology> dim_t program_id() noexcept {
    return program_id_getter<Topology>::id();
}

bool get_profiler_option() noexcept;

template <topology Scope = (topology)(topology_levels - 1)>
auto program_ids() noexcept {
    auto f = []<size_t... Is>(std::index_sequence<Is...>) {
        return make_shape(program_id<static_cast<topology>(Is)>()...);
    };
    return f(std::make_index_sequence<static_cast<size_t>(Scope) + 1>());
}

template <topology Scope> class topology_synchronizer;

template <topology Scope = (topology)0> void topology_synchronize() noexcept {
    topology_synchronizer<Scope>::synchronize();
}
} // namespace nncase::ntt::distributed
