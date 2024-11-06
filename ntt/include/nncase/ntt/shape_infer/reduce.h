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
template <class Index, class Shape>
constexpr size_t reduced_index_by_shape_dim(const Index &src_index,
                                            const Shape &shape, size_t axis) {
    const auto dims_ext = src_index.rank() - shape.rank();
    return src_index[axis + dims_ext] >= shape[axis]
               ? 0
               : src_index[axis + dims_ext];
}

template <class Index, class Shape, class Axes>
struct ranked_reduced_index_by_shape_impl;

template <class Index, class Shape, size_t... Axes>
struct ranked_reduced_index_by_shape_impl<Index, Shape,
                                          std::index_sequence<Axes...>> {
    using type = ranked_shape<Shape::rank()>;

    static constexpr type value(const Index &src_index, const Shape &shape) {
        return type{reduced_index_by_shape_dim(src_index, shape, Axes)...};
    }
};

template <class Index, class Shape, class Axes>
struct fixed_reduced_index_by_shape_impl;

template <class Index, class Shape, size_t... Axes>
struct fixed_reduced_index_by_shape_impl<Index, Shape,
                                         std::index_sequence<Axes...>> {
    using type =
        fixed_shape<reduced_index_by_shape_dim(Index{}, Shape{}, Axes)...>;

    static constexpr type value(const Index &, const Shape &) { return type{}; }
};

template <class Index, class Shape>
struct reduced_index_by_shape_impl
    : ranked_reduced_index_by_shape_impl<
          Index, Shape, std::make_index_sequence<Shape::rank()>> {};

template <size_t... Indices, size_t... Dims>
struct reduced_index_by_shape_impl<fixed_shape<Indices...>,
                                   fixed_shape<Dims...>>
    : fixed_reduced_index_by_shape_impl<
          fixed_shape<Indices...>, fixed_shape<Dims...>,
          std::make_index_sequence<sizeof...(Dims)>> {};
} // namespace detail

template <class Index, class Shape>
constexpr auto reduced_index_by_shape(const Index &src_index,
                                      const Shape &shape) {
    return detail::reduced_index_by_shape_impl<Index, Shape>::value(src_index,
                                                                    shape);
}

} // namespace nncase::ntt::shape_infer
