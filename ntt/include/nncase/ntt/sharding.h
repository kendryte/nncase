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
#include "nncase/ntt/arch/cpu/topology.h"
#include "primitive_ops.h"
#include "shape.h"
#include <cstddef>
#include <tuple>

namespace nncase::ntt::distributed {
namespace shard_policy {
// Broadcast
struct B {};

// Partial
template <reduce_op ReduceOp> struct P {
    static constexpr ntt::reduce_op reduce_op = ReduceOp;
};

// Implicit
struct I {
    template <class Mesh>
    static constexpr size_t local_dim(size_t global_dim) noexcept {
        return global_dim;
    }

    template <class Mesh>
    static constexpr size_t global_offset(
        size_t /* global_dim */,
        const typename Mesh::index_type & /* shard_index */) noexcept {
        return 0;
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

    template <class Mesh>
    static constexpr size_t
    global_offset(size_t global_dim,
                  const typename Mesh::index_type &shard_index) noexcept {
        using submesh_shape = fixed_shape<Mesh::shape_type::at(Axes)...>;
        using submesh_strides = default_strides_t<submesh_shape>;
        ranked_shape<submesh_shape::rank()> submesh_index{
            shard_index.at(Axes)...};
        auto submesh_linear_offset =
            ntt::linear_offset(submesh_index, submesh_strides{});
        auto local_dim = S::local_dim<Mesh>(global_dim);
        return submesh_linear_offset * local_dim;
    }
};
} // namespace shard_policy

template <topology Scope, size_t... Dims> struct mesh {
    using index_type = ranked_shape<sizeof...(Dims)>;
    using shape_type = fixed_shape<Dims...>;
    using strides_type = default_strides_t<shape_type>;
    using program_id_type = ranked_shape<static_cast<size_t>(Scope) + 1>;

    static_assert(shape_type::length() == topology_size<Scope>(),
                  "Invalid mesh shape.");

    static constexpr topology scope = Scope;

    static constexpr program_id_type
    remote_program_id(index_type index) noexcept;

    static constexpr index_type
    index_from_program_id(program_id_type program_id) noexcept;

    static constexpr index_type local_index() noexcept {
        return index_from_program_id(program_ids<Scope>());
    }
};

template <class Mesh, class ImplicitPolicy, class... AxisPolicies>
struct sharding {
    using mesh_type = Mesh;
    using implicit_policy_type = ImplicitPolicy;
    using axis_policy_type = std::tuple<AxisPolicies...>;

    static constexpr size_t axis_policies_size = sizeof...(AxisPolicies);

    static constexpr size_t rank() noexcept { return sizeof...(AxisPolicies); }

    template <class GlobalShape>
    static constexpr ranked_shape<rank()>
    global_offset(const GlobalShape &global_shape,
                  const typename Mesh::index_type &shard_index) noexcept {
        auto get_dim = [&]<size_t Axis> {
            return std::tuple_element_t<Axis, axis_policy_type>::
                template global_offset<Mesh>(global_shape.at(Axis),
                                             shard_index);
        };
        auto get_all_dims = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return ranked_shape<rank()> {
                get_dim.template operator()<Is>()...
            };
        };
        return get_all_dims(std::make_index_sequence<rank()>{});
    }
};

namespace detail {
template <class Mesh, topology Scope>
constexpr size_t get_submesh_rank() noexcept;

template <class Mesh, topology Scope>
constexpr size_t get_submesh_end() noexcept {
    if constexpr (static_cast<size_t>(Scope) == topology_levels - 1) {
        return Mesh::shape_type::rank();
    } else {
        constexpr auto next_topology =
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
    static_assert(GlobalShape::rank() == Sharding::axis_policies_size,
                  "Invalid sharding.");

    auto local_dim = shape.at(Axis);
    return std::get<Axis>(typename Sharding::axis_policy_type{})
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

template <class Mesh, topology Topology>
constexpr size_t
mesh_index_from_program_id(ranked_shape<Mesh::shape_type::rank()> &index,
                           size_t index_offset, size_t program_id) noexcept {
    auto submesh_rank = get_submesh_rank<Mesh, Topology>();
    if (submesh_rank) {
        auto axis = get_submesh_start<Mesh, Topology>();
        for (size_t i = 0; i < submesh_rank; i++) {
            auto divider = Mesh::strides_type::at(i + axis);
            index.at(i + index_offset) = program_id / divider;
            program_id = program_id % divider;
        }
    }
    return submesh_rank;
}

template <class Mesh, size_t... TopologyIndexes>
constexpr typename Mesh::program_id_type
program_ids_in_mesh(ranked_shape<Mesh::shape_type::rank()> index,
                    std::index_sequence<TopologyIndexes...>) noexcept {
    return typename Mesh::program_id_type{
        program_id_in_mesh<Mesh, (topology)TopologyIndexes>(index)...};
}

template <class Mesh, size_t... TopologyIndexes>
constexpr typename Mesh::index_type
mesh_index_from_program_id(typename Mesh::program_id_type program_id,
                           std::index_sequence<TopologyIndexes...>) noexcept {
    typename Mesh::index_type index{};
    size_t index_offset = 0;
    auto f = [&]<size_t TopologyIndex> {
        index_offset +=
            mesh_index_from_program_id<Mesh, (topology)TopologyIndex>(
                index, index_offset, program_id.at(TopologyIndex));
    };
    (f.template operator()<TopologyIndexes>(), ...);
    return index;
}
} // namespace detail

template <topology Scope, size_t... Dims>
constexpr auto mesh<Scope, Dims...>::remote_program_id(
    index_type index) noexcept -> program_id_type {
    return detail::program_ids_in_mesh<mesh>(
        index, std::make_index_sequence<program_id_type::rank()>{});
}

template <topology Scope, size_t... Dims>
constexpr auto mesh<Scope, Dims...>::index_from_program_id(
    program_id_type program_id) noexcept -> index_type {
    return detail::mesh_index_from_program_id<mesh>(
        program_id, std::make_index_sequence<program_id_type::rank()>{});
}
} // namespace nncase::ntt::distributed
