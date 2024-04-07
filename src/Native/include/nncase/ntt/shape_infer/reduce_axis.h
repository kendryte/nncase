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
template <size_t Axis, size_t... Dims, size_t... Ints>
inline constexpr auto
reduced_shape_by_axis_impl(const fixed_shape<Dims...> shape,
                           const std::index_sequence<Ints...>) {
    return fixed_shape<(Ints == Axis ? 1 : shape.at(Ints))...>{};
}

template <size_t... Dims, size_t... Ints>
inline constexpr auto reduced_shape_by_axes_impl(
    const fixed_shape<Dims...> shape, [[maybe_unused]] const fixed_shape<>,
    [[maybe_unused]] const std::index_sequence<Ints...>) {
    return shape;
}

template <size_t AxesFirst, size_t... AxesRest, size_t... Dims, size_t... Ints>
inline constexpr auto
reduced_shape_by_axes_impl(const fixed_shape<Dims...> shape,
                           const fixed_shape<AxesFirst, AxesRest...>,
                           const std::index_sequence<Ints...> ints) {
    return reduced_shape_by_axes_impl(
        reduced_shape_by_axis_impl<AxesFirst>(shape, ints),
        fixed_shape<AxesRest...>{}, ints);
}

} // namespace detail

/**
 * @brief shape[axis] == 1
 *
 * @tparam Axis reduced dim.
 * @param shape input shape.
 * @return changed shape.
 */
template <size_t Axis, size_t... Dims>
inline constexpr auto reduced_shape_by_axis(const fixed_shape<Dims...> shape) {
    return detail::reduced_shape_by_axis_impl<Axis>(
        shape, std::make_index_sequence<sizeof...(Dims)>{});
}

/**
 * @brief [shape[axis] = 1 for axis in axes]
 *
 * @tparam Axes
 * @tparam Dims
 * @param shape
 * @param axes
 * @return constexpr auto
 */
template <size_t... Axes, size_t... Dims>
inline constexpr auto reduced_shape_by_axes(const fixed_shape<Dims...> shape,
                                            const fixed_shape<Axes...> axes) {
    return detail::reduced_shape_by_axes_impl(
        shape, axes, std::make_index_sequence<sizeof...(Dims)>{});
}
} // namespace nncase::ntt::shape_infer