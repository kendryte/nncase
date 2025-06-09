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
#include "../dimension.h"
#include "../shape.h"
#include "mesh.h"
#include "topology.h"
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>

namespace nncase::ntt::distributed {
template <class T>
concept Sharding = requires(T t) {
    typename T::mesh_type;
    typename T::axis_policies_type;
};

namespace shard_policy {
// Broadcast
struct broadcast {
    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto
    global_offset([[maybe_unused]] const TDim &global_dim,
                  [[maybe_unused]] const TShardIndex &shard_index) noexcept {
        return dim_zero;
    }

    template <class Mesh, Dimension TDim>
    static constexpr auto
    try_shard_dim_without_shard_index(const TDim &global_dim) noexcept {
        return std::make_optional(global_dim);
    }

    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto
    shard_dim(const TDim &global_dim,
              [[maybe_unused]] const TShardIndex &shard_index) noexcept {
        return global_dim;
    }
};

inline constexpr broadcast B;

// Split
template <size_t... Axes> struct split {
    static constexpr auto axes = fixed_shape_v<Axes...>;
    static constexpr auto divider = axes.length();

    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto
    global_offset(const TDim &global_dim,
                  const TShardIndex &shard_index) noexcept {
        constexpr auto submesh_shape =
            fixed_shape_v<Mesh::shape[fixed_dim_v<Axes>]...>;
        auto subshard_index = make_shape(shard_index[fixed_dim_v<Axes>]...);
        auto submesh_linear_offset =
            ntt::linear_offset(subshard_index, submesh_shape);
        auto max_offset = max_shard_dim(global_dim) * submesh_linear_offset;
        return ntt::min(max_offset, global_dim);
    }

    template <class Mesh, Dimension TDim>
    static constexpr auto
    try_shard_dim_without_shard_index(const TDim &global_dim) noexcept {
        if constexpr (global_dim % divider == 0) {
            return std::make_optional(global_dim / divider);
        } else {
            return std::nullopt;
        }
    }

    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto
    shard_dim(const TDim &global_dim,
              [[maybe_unused]] const TShardIndex &shard_index) noexcept {
        constexpr auto shard_dim_v =
            try_shard_dim_without_shard_index<Mesh>(global_dim);
        if constexpr (shard_dim_v.has_value()) {
            return shard_dim_v.value();
        } else {
            auto offset = global_offset<Mesh>(global_dim, shard_index);
            return ntt::min(global_dim - offset, max_shard_dim(global_dim));
        }
    }

    template <Dimension TDim>
    static constexpr auto max_shard_dim(const TDim &global_dim) noexcept {
        return ntt::ceil_div(global_dim, divider);
    }
};

template <size_t... Axes> constexpr auto S() noexcept {
    return split<Axes...>{};
}
} // namespace shard_policy

template <class Policy> struct is_split_shard_policy : std::false_type {};

template <size_t... Axes>
struct is_split_shard_policy<shard_policy::split<Axes...>> : std::true_type {};

template <class Policy>
concept SplitShardPolicy = is_split_shard_policy<Policy>::value;

template <class Mesh, class... AxisPolicies> struct sharding {
    using mesh_type = Mesh;
    using axis_policies_type = std::tuple<AxisPolicies...>;
    using dynamic_offset_t = dynamic_shape_t<sizeof...(AxisPolicies)>;

    static constexpr size_t axis_policies_size = sizeof...(AxisPolicies);

    constexpr sharding(const AxisPolicies &...axis_policies) noexcept
        : axis_policies(axis_policies...) {}

    template <Shape GlobalShape, ShardIndex<Mesh> TShardIndex>
    constexpr auto
    global_offset(const GlobalShape &global_shape,
                  const TShardIndex &shard_index) const noexcept {
        auto get_dim = [&, this]<size_t Axis> {
            return std::get<Axis>(axis_policies)
                .template global_offset<Mesh>(global_shape[fixed_dim_v<Axis>],
                                              shard_index);
        };
        auto get_all_dims = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return make_shape(get_dim.template operator()<Is>()...);
        };
        return get_all_dims(std::make_index_sequence<axis_policies_size>{});
    }

    template <Shape GlobalShape, ShardIndex<Mesh> TShardIndex>
    constexpr auto shard_shape(const GlobalShape &global_shape,
                               const TShardIndex &shard_index) const noexcept {
        auto get_dim = [&, this]<size_t Axis> {
            return std::get<Axis>(axis_policies)
                .template shard_dim<Mesh>(global_shape[fixed_dim_v<Axis>],
                                          shard_index);
        };
        auto get_all_dims = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return make_shape(get_dim.template operator()<Is>()...);
        };
        return get_all_dims(std::make_index_sequence<axis_policies_size>{});
    }

    std::tuple<AxisPolicies...> axis_policies;
};

template <class Mesh, class... AxisPolicies>
constexpr auto make_sharding(const AxisPolicies &...axis_policies) noexcept {
    return sharding<Mesh, AxisPolicies...>(axis_policies...);
}

namespace detail {
template <size_t Axis, class Sharding, class GlobalShape>
constexpr auto get_local_shard_dim(const Sharding &sharding,
                                   const GlobalShape &shape) noexcept {
    static_assert(GlobalShape::rank() == Sharding::axis_policies_size,
                  "Invalid sharding.");

    auto local_dim = shape.at(Axis);
    return std::get<Axis>(sharding.axis_policies)
        .template local_dim<typename Sharding::mesh_type>(local_dim);
}

template <class Sharding, class GlobalShape, size_t... Ids>
constexpr bool is_divisible(const Sharding &sharding, const GlobalShape &shape,
                            std::index_sequence<Ids...>) noexcept {
    return ((std::get<Ids>(sharding.axis_policies)
                 .template is_divisible<typename Sharding::mesh_type>(
                     shape.at(Ids))) &&
            ...);
}

template <class Sharding, Shape GlobalShape>
constexpr auto local_shard_shape(const Sharding &sharding,
                                 const GlobalShape &shape) noexcept {
    auto get_dims = [&]<size_t... Axes>(std::index_sequence<Axes...>) {
        return make_ranked_shape(get_local_shard_dim<Axes>(sharding, shape)...);
    };
    return get_dims(std::make_index_sequence<GlobalShape::rank()>{});
}
} // namespace detail
} // namespace nncase::ntt::distributed
