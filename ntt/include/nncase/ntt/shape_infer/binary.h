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
#include <cassert>

namespace nncase::ntt::shape_infer {
namespace detail {
template <class ShapeA, class ShapeB>
constexpr size_t binary_output_dim(const ShapeA &shape_a, const ShapeB &shape_b,
                                   size_t axis) {
    const auto dest_dims = std::max(shape_a.rank(), shape_b.rank());
    const auto in_a_ext = dest_dims - shape_a.rank();
    const auto in_b_ext = dest_dims - shape_b.rank();

    const auto in_a_dim = (int32_t)axis - (int32_t)in_a_ext;
    const auto in_b_dim = (int32_t)axis - (int32_t)in_b_ext;

    const auto in_a = in_a_dim < 0 ? 1 : shape_a[in_a_dim];
    const auto in_b = in_b_dim < 0 ? 1 : shape_b[in_b_dim];

    if (in_a == in_b) {
        return in_a;
    } else if (in_a == 1) {
        return in_b;
    } else if (in_b == 1) {
        return in_a;
    } else {
        assert(!"inputs are not compatible to broadcast");
        return -1;
    }
}

template <class ShapeA, class ShapeB, class Axes>
struct ranked_binary_output_shape_impl;

template <class ShapeA, class ShapeB, size_t... Axes>
struct ranked_binary_output_shape_impl<ShapeA, ShapeB,
                                       std::index_sequence<Axes...>> {
    using type = ranked_shape<std::max(ShapeA::rank(), ShapeB::rank())>;

    static constexpr type value(const ShapeA &shape_a, const ShapeB &shape_b) {
        return type{binary_output_dim(shape_a, shape_b, Axes)...};
    }
};

template <class ShapeA, class ShapeB, class Axes>
struct fixed_binary_output_shape_impl;

template <class ShapeA, class ShapeB, size_t... Axes>
struct fixed_binary_output_shape_impl<ShapeA, ShapeB,
                                      std::index_sequence<Axes...>> {
    using type = fixed_shape<binary_output_dim(ShapeA{}, ShapeB{}, Axes)...>;

    static constexpr type value(const ShapeA &, const ShapeB &) {
        return type{};
    }
};

template <class ShapeA, class ShapeB>
struct binary_output_shape_impl
    : ranked_binary_output_shape_impl<
          ShapeA, ShapeB,
          std::make_index_sequence<std::max(ShapeA::rank(), ShapeB::rank())>> {
};

template <size_t... DimsA, size_t... DimsB>
struct binary_output_shape_impl<fixed_shape<DimsA...>, fixed_shape<DimsB...>>
    : fixed_binary_output_shape_impl<fixed_shape<DimsA...>,
                                     fixed_shape<DimsB...>,
                                     std::make_index_sequence<std::max(
                                         sizeof...(DimsA), sizeof...(DimsB))>> {
};
} // namespace detail

template <class ShapeA, class ShapeB>
constexpr auto binary_output_shape(const ShapeA &shape_a,
                                   const ShapeB &shape_b) {
    return detail::binary_output_shape_impl<ShapeA, ShapeB>::value(shape_a,
                                                                   shape_b);
}

} // namespace nncase::ntt::shape_infer
