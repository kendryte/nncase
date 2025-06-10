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
#include "../tensor_traits.h"
#include "topology.h"

namespace nncase::ntt::distributed {
template <topology RemoteScope> struct remote_tensor_constructor;

template <topology RemoteScope, ScalarOrVector T, class Sharding,
          Shape GlobalShape, Shape ShardIndex, Strides TStrides>
constexpr auto make_remote_tensor(T *data, const Sharding &sharding,
                                  const GlobalShape &global_shape,
                                  const TStrides &strides,
                                  const ShardIndex &shard_index) noexcept {
    using mesh_type = typename Sharding::mesh_type;
    constexpr auto tensor_scope = mesh_type::scope;
    static_assert(RemoteScope <= tensor_scope,
                  "Remote scope must be higher than or equal to tensor scope.");
    constexpr auto cross_topology_count = static_cast<size_t>(tensor_scope) -
                                          static_cast<size_t>(RemoteScope) + 1;
    constexpr auto no_cross_topology_count = static_cast<size_t>(RemoteScope);
    static_assert(
        shard_index.rank() == cross_topology_count,
        "Shard index must have the same rank as cross topology count.");

    const auto local_shard_index = mesh_type::local_shard_index();
    const auto local_program_ids =
        program_ids_from_shard_index<mesh_type>(local_shard_index);
    const auto no_cross_shard_index =
        local_shard_index.template slice<0, no_cross_topology_count>();
    const auto remote_shard_index = no_cross_shard_index.concat(shard_index);
    const auto remote_program_ids =
        program_ids_from_shard_index<mesh_type>(remote_shard_index);
    const auto remote_shard_shape =
        sharding.shard_shape(global_shape, remote_shard_index);
    return remote_tensor_constructor<RemoteScope>()(data, remote_shard_shape,
                                                    strides, local_program_ids,
                                                    remote_program_ids);
}
} // namespace nncase::ntt::distributed
