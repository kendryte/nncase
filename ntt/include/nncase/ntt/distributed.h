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
#include "nncase/ntt/shape.h"
#include "primitive_ops.h"
#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <utility>

#ifdef NNCASE_CPU_MODULE
#include <topology_def.h>
#endif

namespace nncase::ntt {
inline constexpr size_t topology_levels =
    static_cast<size_t>(topology::count__);

#ifndef NNCASE_CPU_MODULE
constexpr size_t program_dim(topology /* topo */) noexcept { return 1; }
#else
constexpr size_t program_dim(topology topo) noexcept {
    auto index =
        static_cast<size_t>(topo) - (topology_levels - topology_dims.size());
    return topology_dims[index];
}
#endif

template <topology Topology> struct program_id_getter {
    static size_t id() noexcept;
};

template <topology Topology> size_t program_id() noexcept {
    return program_id_getter<Topology>::id();
}

namespace dist_policy {
// Broadcast
struct B {
    static constexpr size_t local_dim(size_t global_dim,
                                      size_t /* topology_dim */,
                                      size_t /* axis */) noexcept {
        return global_dim;
    }
};

// Split
template <size_t Axis> struct S {
    static constexpr size_t axis = Axis;

    static constexpr size_t local_dim(size_t global_dim, size_t topology_dim,
                                      size_t axis) noexcept {
        return axis == Axis ? ntt::ceil_div(global_dim, topology_dim)
                            : global_dim;
    }
};

// Partial
struct P {
    static constexpr size_t local_dim(size_t global_dim,
                                      size_t /* topology_dim */,
                                      size_t /* axis */) noexcept {
        return global_dim;
    }
};
} // namespace dist_policy

template <class... TPolicies> struct dist {
    static constexpr std::tuple<TPolicies...> policies = {TPolicies{}...};
    static constexpr size_t size = sizeof...(TPolicies);
};

namespace detail {
template <class TDist, topology Scope, class GlobalShape>
constexpr size_t get_local_dim(GlobalShape shape, size_t axis) noexcept {
    auto local_dim = shape.at(axis);
    auto cnt_topology =
        static_cast<topology>(static_cast<size_t>(Scope) + 1 - TDist::size);
    auto apply_policy = [&](auto policy) {
        local_dim =
            policy.local_dim(local_dim, program_dim(cnt_topology), axis);
        cnt_topology =
            static_cast<topology>(static_cast<size_t>(cnt_topology) + 1);
    };
    std::apply([&](auto... policies) { (apply_policy(policies), ...); },
               TDist::policies);
    return local_dim;
}

template <class TDist, topology Scope, class GlobalShape, size_t... Axes>
constexpr auto get_fixed_local_dim(GlobalShape,
                                   std::index_sequence<Axes...>) noexcept {
    return fixed_shape<get_local_dim<TDist, Scope>(GlobalShape{}, Axes)...>{};
}

template <class GlobalShape, class TDist, topology Scope>
struct local_shape_type {
    using type = ranked_shape<GlobalShape::rank()>;
};

template <size_t... Dims, class TDist, topology Scope>
struct local_shape_type<fixed_shape<Dims...>, TDist, Scope> {
    using type = decltype(get_fixed_local_dim<TDist, Scope>(
        fixed_shape<Dims...>{}, std::make_index_sequence<sizeof...(Dims)>{}));
};
} // namespace detail

template <
    class T, class GlobalShape, class TDist, topology Scope,
    class LocalStrides = default_strides_t<
        typename detail::local_shape_type<GlobalShape, TDist, Scope>::type>>
class dist_tensor_view
    : public tensor_view<
          T, typename detail::local_shape_type<GlobalShape, TDist, Scope>::type,
          LocalStrides> {
  public:
    using local_tensor_type = tensor_view<
        T, typename detail::local_shape_type<GlobalShape, TDist, Scope>::type,
        LocalStrides>;

    using local_tensor_type::local_tensor_type;

    local_tensor_type &local() noexcept {
        return static_cast<local_tensor_type &>(*this);
    }

    const local_tensor_type &local() const noexcept {
        return static_cast<local_tensor_type &>(*this);
    }
};
} // namespace nncase::ntt
