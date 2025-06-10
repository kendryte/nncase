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
#include "../shape.h"
#include "topology.h"

namespace nncase::ntt::distributed {
template <class T, class Mesh>
concept ShardIndex = Shape<T> && bool(T::rank() == Mesh::rank());

template <topology Scope, size_t... Dims> struct mesh {
    using dynamic_shard_index_type = dynamic_shape_t<sizeof...(Dims)>;
    using scoped_dynamic_program_ids_type = dynamic_program_ids_t<Scope>;

    static constexpr topology scope = Scope;
    static constexpr auto shape = fixed_shape_v<Dims...>;
    static constexpr auto rank() noexcept { return shape.rank(); }

    static_assert(shape.length() == topology_up_size<Scope>(),
                  "Invalid mesh shape.");

    template <Shape GlobalShape, ShardIndex<mesh<Scope, Dims...>> TShardIndex>
    static constexpr auto
    shard_shape_by_index(const GlobalShape &global_shape,
                         const TShardIndex &shard_index) noexcept;

    template <ShardIndex<mesh<Scope, Dims...>> TShardIndex>
    static constexpr auto
    program_ids_from_index(const TShardIndex &shard_index) noexcept;

    template <ScopedProgramIds<Scope> TProgramIds>
    static constexpr auto
    index_from_program_ids(const TProgramIds &program_ids) noexcept;

    static constexpr auto local_program_ids() noexcept {
        return program_ids<Scope>();
    }

    static constexpr auto local_index() noexcept {
        return index_from_program_ids(local_program_ids());
    }
};

namespace detail {
template <class Mesh, topology Scope>
constexpr size_t get_submesh_rank() noexcept;

template <class Mesh, topology Scope>
constexpr size_t get_submesh_end() noexcept {
    if constexpr (static_cast<size_t>(Scope) == topology_levels - 1) {
        return Mesh::rank();
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
            size *= Mesh::shape.at(--index);
            if (size == program_dim<Scope>())
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

template <class Mesh, topology Topology, ShardIndex<Mesh> TShardIndex>
constexpr auto
program_id_from_shard_index(const TShardIndex &shard_index) noexcept {
    constexpr auto submesh_rank = get_submesh_rank<Mesh, Topology>();
    if constexpr (submesh_rank) {
        constexpr auto axis = fixed_dim_v<get_submesh_start<Mesh, Topology>()>;
        dim_t id = shard_index[axis];
        for (size_t i = 0; i < submesh_rank - 1; i++) {
            auto next_dim = Mesh::shape[axis + 1_dim];
            id = id * next_dim + shard_index[axis + 1_dim];
            axis++;
        }
        return id;
    } else {
        return dim_zero;
    }
}

template <class Mesh, topology Topology, Dimension TProgramId>
constexpr auto
subshard_index_from_program_id(const TProgramId &program_id) noexcept {
    constexpr auto submesh_rank = get_submesh_rank<Mesh, Topology>();
    if constexpr (submesh_rank) {
        constexpr auto submesh_start = get_submesh_start<Mesh, Topology>();
        auto submesh_shape =
            Mesh::shape.template slice<submesh_start, submesh_rank>();
        return unravel_index<shape_t>(program_id, submesh_shape);

    } else {
        return shape_t<>{};
    }
}

template <class Mesh, ShardIndex<Mesh> TShardIndex, size_t... TopologyIndexes>
constexpr auto
program_ids_from_shard_index(const TShardIndex &shard_index,
                             std::index_sequence<TopologyIndexes...>) noexcept {
    return make_shape(
        program_id_from_shard_index<Mesh, (topology)TopologyIndexes>(
            shard_index)...);
}

template <class Mesh, ScopedProgramIds<Mesh::scope> TProgramIds,
          size_t... TopologyIndexes>
static constexpr auto
shard_index_from_program_ids(const TProgramIds &program_ids,
                             std::index_sequence<TopologyIndexes...>) noexcept {
    return fixed_shape_v<(dim_t)TopologyIndexes...>.aggregate(
        shape_t<>{},
        [&](auto last_index, auto topo, [[maybe_unused]] auto axis) {
            auto program_id = program_ids.at(topo);
            auto subshard_index =
                subshard_index_from_program_id<Mesh, (topology)topo.value>(
                    program_id);
            return last_index.concat(subshard_index);
        });
}
} // namespace detail

template <topology Scope, size_t... Dims>
template <ShardIndex<mesh<Scope, Dims...>> TShardIndex>
constexpr auto mesh<Scope, Dims...>::program_ids_from_index(
    const TShardIndex &shard_index) noexcept {
    return detail::program_ids_from_shard_index<mesh<Scope, Dims...>>(
        shard_index, std::make_index_sequence<topology_levels>{});
}

template <topology Scope, size_t... Dims>
template <ScopedProgramIds<Scope> TProgramIds>
constexpr auto mesh<Scope, Dims...>::index_from_program_ids(
    const TProgramIds &program_ids) noexcept {
    return detail::shard_index_from_program_ids<mesh<Scope, Dims...>>(
        program_ids, std::make_index_sequence<topology_levels>{});
}
} // namespace nncase::ntt::distributed
