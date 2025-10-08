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
#include "dimension.h"
#include "loop.h"
#include "nncase/ntt/compiler_defs.h"
#include "nncase/ntt/tensor_traits.h"
#include "shape.h"
#include <cstddef>
#include <tuple>
#include <utility>

namespace nncase::ntt {
namespace detail {
template <size_t Axis, class Shape, FixedShape TTile, class Offsets,
          class Callable, class... Strides>
NTT_ALWAYS_INLINE constexpr void
apply_impl(dynamic_shape_t<Shape::rank()> &index, Offsets offsets,
           const Shape &shape, const TTile &tile, Callable &&callable,
           const std::tuple<Strides...> &strides) {
    auto call = [&]<size_t... I>(std::index_sequence<I...>) {
        if constexpr (sizeof...(Strides)) {
            callable(index, offsets[fixed_dim_v<I>]...);
        } else {
            callable(index);
        }
    };
    auto &dim = index[fixed_dim_v<Axis>];
    for (dim = 0; dim < shape[fixed_dim_v<Axis>];
         dim += tile[fixed_dim_v<Axis>]) {
        if constexpr (Axis == Shape::rank() - 1) {
            call(std::make_index_sequence<sizeof...(Strides)>{});
        } else {
            apply_impl<Axis + 1>(index, offsets, shape, tile,
                                 std::forward<Callable>(callable), strides);
        }
        ntt::loop<sizeof...(Strides)>([&](auto i) {
            offsets[i] += std::get<i>(strides)[fixed_dim_v<Axis>] *
                          tile[fixed_dim_v<Axis>];
        });
    }
}
} // namespace detail

template <Shape TShape, class Callable, Strides... TStrides>
NTT_ALWAYS_INLINE constexpr void apply(const TShape &shape, Callable &&callable,
                                       const TStrides &...strides) {
    if constexpr (TShape::rank()) {
        dynamic_shape_t<TShape::rank()> index{};
        detail::apply_impl<0>(index, make_repeat_shape<sizeof...(TStrides)>(0),
                              shape, make_ones_shape<TShape::rank()>(),
                              std::forward<Callable>(callable),
                              std::forward_as_tuple(strides...));
    } else {
        if constexpr (sizeof...(TStrides)) {
            callable(fixed_shape_v<>, (strides, (dim_t)0)...);
        } else {
            callable(fixed_shape_v<>);
        }
    }
}

template <Shape TShape, FixedShape TTile, class Callable, Strides... TStrides>
NTT_ALWAYS_INLINE constexpr void
apply_tiled(const TShape &shape, const TTile &tile, Callable &&callable,
            const TStrides &...strides) {
    if constexpr (TShape::rank()) {
        dynamic_shape_t<TShape::rank()> index{};
        detail::apply_impl<0>(index, make_repeat_shape<sizeof...(TStrides)>(0),
                              shape, tile, std::forward<Callable>(callable),
                              std::forward_as_tuple(strides...));
    } else {
        if constexpr (sizeof...(TStrides)) {
            callable(fixed_shape_v<>, (strides, (dim_t)0)...);
        } else {
            callable(fixed_shape_v<>);
        }
    }
}
} // namespace nncase::ntt
