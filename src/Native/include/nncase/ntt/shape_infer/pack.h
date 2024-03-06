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

template <size_t Lanes, size_t Axis, size_t OffSet, class Indices>
static constexpr size_t packed_index_by_shape_dim(const Indices &out_indices,
                                                  const size_t i) {
    if (i == Axis) {
        return out_indices[i] * Lanes + OffSet;
    }

    return out_indices[i];
}

template <class Indices, size_t Lanes, size_t Axis, size_t OffSet, class Axes>
struct ranked_packed_index_by_shape_impl;

template <class Indices, size_t Lanes, size_t Axis, size_t OffSet,
          size_t... Axes>
struct ranked_packed_index_by_shape_impl<Indices, Lanes, Axis, OffSet,
                                         std::index_sequence<Axes...>> {
    using type = ranked_shape<Indices::rank()>;

    static constexpr type value(const Indices &out_indices) {
        return type{packed_index_by_shape_dim<Lanes, Axis, OffSet>(out_indices,
                                                                   Axes)...};
    }
};

template <class Indices, size_t Lanes, size_t Axis, size_t OffSet, class Axes>
struct fixed_packed_index_by_shape_impl;

template <class Indices, size_t Lanes, size_t Axis, size_t OffSet,
          size_t... Axes>
struct fixed_packed_index_by_shape_impl<Indices, Lanes, Axis, OffSet,
                                        std::index_sequence<Axes...>> {
    using type = fixed_shape<packed_index_by_shape_dim<Lanes, Axis, OffSet>(
        Indices{}, Axes)...>;

    static constexpr type value(const Indices &) { return type{}; }
};

template <class Indices, size_t Lanes, size_t Axis, size_t OffSet>
struct packed_index_by_shape_impl
    : ranked_packed_index_by_shape_impl<
          Indices, Lanes, Axis, OffSet,
          std::make_index_sequence<Indices::rank()>> {};

template <size_t... Indices, size_t Lanes, size_t Axis, size_t OffSet>
struct packed_index_by_shape_impl<fixed_shape<Indices...>, Axis, Lanes, OffSet>
    : fixed_packed_index_by_shape_impl<
          fixed_shape<Indices...>, Lanes, Axis, OffSet,
          std::make_index_sequence<sizeof...(Indices)>> {};

} // namespace detail


template <size_t Lanes, size_t Axis, size_t OffSet, class Indices>
constexpr auto packed_index_by_shape(const Indices &out_indices) {
    return detail::packed_index_by_shape_impl<Indices, Lanes, Axis,
                                              OffSet>::value(out_indices);
}

} // namespace nncase::ntt::shape_infer
