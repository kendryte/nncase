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

namespace nncase::ntt::shape_infer {
namespace detail {
template <size_t Axis, Shape TShape, size_t... Ints>
inline constexpr auto
reduced_shape_by_axis_impl([[maybe_unused]] const TShape &shape,
                           std::index_sequence<Ints...>) {
    return make_shape((Ints == Axis ? dim_one : shape[fixed_dim_v<Ints>])...);
}

template <Shape TShape, Shape TAxes, size_t... Ints>
inline constexpr auto
reduced_shape_by_axes_impl([[maybe_unused]] const TShape &shape,
                           const TAxes &axes, std::index_sequence<Ints...>) {
    return make_shape((axes.contains(fixed_dim_v<Ints>)
                           ? dim_one
                           : shape[fixed_dim_v<Ints>])...);
}

} // namespace detail

/**
 * @brief shape[axis] == 1
 *
 * @tparam Axis reduced dim.
 * @param shape input shape.
 * @return changed shape.
 */
template <size_t Axis, Shape TShape>
constexpr auto reduced_shape_by_axis(const TShape &shape) {
    return detail::reduced_shape_by_axis_impl<Axis>(
        shape, std::make_index_sequence<TShape::rank()>());
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
template <Shape TShape, Shape TAxes>
constexpr auto reduced_shape_by_axes(const TShape &shape, const TAxes &axes) {
    return detail::reduced_shape_by_axes_impl(
        shape, axes, std::make_index_sequence<TShape::rank()>{});
}
} // namespace nncase::ntt::shape_infer
