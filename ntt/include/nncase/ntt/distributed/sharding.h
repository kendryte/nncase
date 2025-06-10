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

    template <class Mesh> static constexpr auto divider() {
        return Mesh::shape.select(axes).length();
    }

    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto
    global_offset(const TDim &global_dim,
                  const TShardIndex &shard_index) noexcept {
        constexpr auto submesh_shape =
            fixed_shape_v<Mesh::shape[fixed_dim_v<Axes>]...>;
        auto subshard_index = make_shape(shard_index[fixed_dim_v<Axes>]...);
        auto submesh_linear_offset =
            ntt::linear_offset(subshard_index, submesh_shape);
        auto max_offset =
            max_shard_dim<Mesh>(global_dim) * submesh_linear_offset;
        return ntt::min(max_offset, global_dim);
    }

    template <class Mesh, Dimension TDim>
    static constexpr auto
    try_shard_dim_without_shard_index(const TDim &global_dim) noexcept {
        const auto remainder = global_dim % divider<Mesh>();
        return ntt::where(remainder == dim_zero, global_dim / divider<Mesh>(),
                          -1_dim);
    }

    template <class Mesh, Dimension TDim, ShardIndex<Mesh> TShardIndex>
    static constexpr auto shard_dim(const TDim &global_dim,
                                    const TShardIndex &shard_index) noexcept {
        const auto shard_dim_v =
            try_shard_dim_without_shard_index<Mesh>(global_dim);
        return ntt::where(shard_dim_v != -1_dim, shard_dim_v, [&] {
            auto offset = global_offset<Mesh>(global_dim, shard_index);
            return ntt::min(global_dim - offset,
                            max_shard_dim<Mesh>(global_dim));
        }());
    }

    template <class Mesh, Dimension TDim>
    static constexpr auto max_shard_dim(const TDim &global_dim) noexcept {
        return ntt::ceil_div(global_dim, divider<Mesh>());
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

    static constexpr auto rank() {
        return fixed_dim_v<sizeof...(AxisPolicies)>;
    }

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
        return get_all_dims(std::make_index_sequence<rank()>{});
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
        return get_all_dims(std::make_index_sequence<rank()>{});
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
    static_assert(GlobalShape::rank() == Sharding::rank(), "Invalid sharding.");

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

template <Sharding TSharding>
constexpr auto mesh_axes_mask_of_split_shard_policies() noexcept {
    using mesh_type = typename TSharding::mesh_type;
    return generate_shape<mesh_type::rank()>([](auto mesh_axis) {
        return make_index_shape<TSharding::rank()>().aggregate(
            dim_zero, [&](auto last_mask, auto axis, auto) {
                using policy_t = std::tuple_element_t<
                    axis, typename TSharding::axis_policies_type>;
                if constexpr (distributed::SplitShardPolicy<policy_t>) {
                    if constexpr (policy_t::axes.contains(mesh_axis)) {
                        return dim_one;
                    } else {
                        return last_mask;
                    }
                } else {
                    return last_mask;
                }
            });
    });
}

template <Sharding TSharding>
constexpr auto mesh_axes_of_non_split_shard_policies() noexcept {
    using mesh_type = typename TSharding::mesh_type;
    constexpr auto axes_mask =
        mesh_axes_mask_of_split_shard_policies<TSharding>();
    return axes_mask.aggregate(
        fixed_shape_v<>, [&](auto last_axes, auto mask, auto mesh_axis) {
            return ntt::where(mask == dim_zero, last_axes.append(mesh_axis),
                              last_axes);
        });
}

template <Sharding TSharding>
constexpr auto tensor_axes_mask_of_split_shard_policies() noexcept {
    using mesh_type = typename TSharding::mesh_type;
    return generate_shape<TSharding::rank()>([](auto axis) {
        using policy_t =
            std::tuple_element_t<axis, typename TSharding::axis_policies_type>;
        if constexpr (distributed::SplitShardPolicy<policy_t>) {
            return dim_one;
        } else {
            return dim_zero;
        }
    });
}

template <Sharding TSharding>
constexpr auto tensor_axes_of_non_split_shard_policies() noexcept {
    constexpr auto axes_mask =
        tensor_axes_mask_of_split_shard_policies<TSharding>();
    return axes_mask.aggregate(fixed_shape_v<>, [&](auto last_axes, auto mask,
                                                    auto axis) {
        return ntt::where(mask == dim_zero, last_axes.append(axis), last_axes);
    });
}
} // namespace detail
} // namespace nncase::ntt::distributed
