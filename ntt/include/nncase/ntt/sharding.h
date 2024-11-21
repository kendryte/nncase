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
#include "distributed.h"
#include "primitive_ops.h"
#include "shape.h"

namespace nncase::ntt::distributed {
namespace dist_policy {
// Broadcast
struct B {
    template <class Mesh>
    static constexpr size_t local_dim(size_t global_dim) noexcept {
        return global_dim;
    }
};

// Split
template <size_t... Axes> struct S {
    using axes_type = fixed_shape<Axes...>;

    template <class Mesh>
    static constexpr size_t local_dim(size_t global_dim) noexcept {
        auto divider = (1 * ... * Mesh::shape_type::at(Axes));
        return ntt::ceil_div(global_dim, divider);
    }
};

// Partial
struct P {
    template <class Mesh>
    static constexpr size_t local_dim(size_t global_dim) noexcept {
        return global_dim;
    }
};
} // namespace dist_policy

template <topology Scope, size_t... Dims> struct mesh {
    using shape_type = fixed_shape<Dims...>;

    static_assert(shape_type::length() == topology_size<Scope>(),
                  "Invalid mesh shape.");

    static constexpr ranked_shape<topology_levels>
    remote_program_id(ranked_shape<shape_type::rank()> index) noexcept;
};

template <class Mesh, class... Policies> struct sharding {
    using mesh_type = Mesh;

    static constexpr std::tuple<Policies...> policies = {Policies{}...};
    static constexpr size_t policies_size = sizeof...(Policies);
};

namespace detail {
template <class Mesh, topology Scope>
constexpr size_t get_submesh_rank() noexcept;

template <class Mesh, topology Scope>
constexpr size_t get_submesh_end() noexcept {
    if (static_cast<size_t>(Scope) == topology_levels - 1) {
        return Mesh::shape_type::rank();
    } else {
        auto next_topology =
            static_cast<topology>(static_cast<size_t>(Scope) + 1);
        return get_submesh_end<Mesh, next_topology>() -
               get_submesh_rank<Mesh, next_topology>();
    }
}

template <class Mesh, topology Scope>
constexpr size_t get_submesh_rank() noexcept {
    auto end = get_submesh_end<Mesh, Scope>();
    if (end) {
        auto index = end;
        size_t size = 1;
        while (true) {
            size *= Mesh::shape_type::at(--index);
            if (size == program_dim(Scope))
                break;
        }
        return end - index;
    }
    return 0;
}

template <class Mesh, topology Scope>
constexpr size_t get_submesh_start() noexcept {
    return get_submesh_end<Mesh, Scope>() - get_submesh_rank<Mesh, Scope>();
}

template <class Sharding, size_t Axis, class GlobalShape>
constexpr size_t get_local_shard_dim(GlobalShape shape) noexcept {
    static_assert(GlobalShape::rank() == Sharding::policies_size,
                  "Invalid sharding.");

    auto local_dim = shape.at(Axis);
    return std::get<Axis>(Sharding::policies)
        .template local_dim<typename Sharding::mesh_type>(local_dim);
}

template <class Sharding, class GlobalShape, size_t... Axes>
constexpr auto
get_fixed_local_shard_dim(GlobalShape, std::index_sequence<Axes...>) noexcept {
    return fixed_shape<get_local_shard_dim<Sharding, Axes>(GlobalShape{})...>{};
}

template <class GlobalShape, class Sharding> struct local_shard_shape_type {
    using type = ranked_shape<GlobalShape::rank()>;
};

template <size_t... Dims, class Sharding>
struct local_shard_shape_type<fixed_shape<Dims...>, Sharding> {
    using type = decltype(get_fixed_local_shard_dim<Sharding>(
        fixed_shape<Dims...>{}, std::make_index_sequence<sizeof...(Dims)>{}));
};

template <class Mesh, topology Topology>
constexpr size_t
program_id_in_mesh(ranked_shape<Mesh::shape_type::rank()> index) noexcept {
    auto submesh_rank = get_submesh_rank<Mesh, Topology>();
    if (submesh_rank) {
        auto axis = get_submesh_start<Mesh, Topology>();
        size_t id = 0;
        for (size_t i = 0; i < submesh_rank - 1; i++) {
            auto next_dim = Mesh::shape_type::at(axis + 1);
            id = id * next_dim + index.at(axis);
            axis++;
        }
        id = id + index.at(axis);
        return id;
    }
    return 0;
}

template <class Mesh, size_t... TopologyIndexes>
constexpr ranked_shape<topology_levels>
program_ids_in_mesh(ranked_shape<Mesh::shape_type::rank()> index,
                    std::index_sequence<TopologyIndexes...>) noexcept {
    return ranked_shape<topology_levels>{
        program_id_in_mesh<Mesh, (topology)TopologyIndexes>(index)...};
}
} // namespace detail

template <topology Scope, size_t... Dims>
constexpr ranked_shape<topology_levels> mesh<Scope, Dims...>::remote_program_id(
    ranked_shape<shape_type::rank()> index) noexcept {
    return detail::program_ids_in_mesh<mesh>(
        index, std::make_index_sequence<topology_levels>{});
}
} // namespace nncase::ntt::distributed
