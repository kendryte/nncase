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
template <class Shape>
constexpr size_t sub_matmul_shape_dim(const Shape &shape, size_t axis) {
    return axis >= Shape::rank() - 2 ? shape[axis] : 1;
}

template <class Shape, class Axes> struct ranked_sub_matmul_shape_impl;

template <class Shape, size_t... Axes>
struct ranked_sub_matmul_shape_impl<Shape, std::index_sequence<Axes...>> {
    using type = ranked_shape<Shape::rank()>;

    static constexpr type value(const Shape &shape) {
        return type{sub_matmul_shape_dim(shape, Axes)...};
    }
};

template <class Shape, class Axes> struct fixed_sub_matmul_shape_impl;

template <class Shape, size_t... Axes>
struct fixed_sub_matmul_shape_impl<Shape, std::index_sequence<Axes...>> {
    using type = fixed_shape<sub_matmul_shape_dim(Shape{}, Axes)...>;

    static constexpr type value(const Shape &) { return type{}; }
};

template <class Shape>
struct sub_matmul_shape_impl
    : ranked_sub_matmul_shape_impl<Shape,
                                   std::make_index_sequence<Shape::rank()>> {};

template <size_t... Dims>
struct sub_matmul_shape_impl<fixed_shape<Dims...>>
    : fixed_sub_matmul_shape_impl<fixed_shape<Dims...>,
                                  std::make_index_sequence<sizeof...(Dims)>> {};
} // namespace detail

template <class Shape> constexpr auto sub_matmul_shape(const Shape &shape) {
    return detail::sub_matmul_shape_impl<Shape>::value(shape);
}

} // namespace nncase::ntt::shape_infer
