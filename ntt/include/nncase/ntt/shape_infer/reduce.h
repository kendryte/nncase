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
#include "nncase/ntt/dimension.h"
#include <type_traits>

namespace nncase::ntt::shape_infer {
namespace detail {
template <size_t Axis, Shape TIndex, Shape TShape>
constexpr auto reduced_index_by_shape_dim(const TIndex &src_index,
                                          const TShape &shape) noexcept {
    const auto dims_ext = src_index.rank() - shape.rank();
    const auto axis_v = fixed_dim_v<Axis>;
    const auto lhs = src_index[axis_v + dims_ext];
    const auto rhs = shape[axis_v];
    if constexpr (FixedDimension<std::decay_t<decltype(lhs)>> &&
                  FixedDimension<std::decay_t<decltype(rhs)>>) {
        return fixed_dim_v<(lhs >= rhs ? 0 : lhs)>;
    } else {
        return lhs >= rhs ? 0 : dim_value(lhs);
    }
}

template <size_t InRank, FixedDimensions ReduceAxes, Shape TOutIndex>
struct reduce_source_begin_index_impl {
    template <size_t Axis, size_t ShrinkedDims, Shape TInIndex>
    constexpr auto operator()(const TInIndex &in_index,
                              const TOutIndex &out_index) noexcept {
        auto [new_dim, new_shrinked_dims] = [&] {
            if constexpr (ReduceAxes{}.contains(fixed_dim_v<Axis>)) {
                return std::make_tuple(dim_zero, fixed_dim_v<ShrinkedDims + 1>);
            } else {
                return std::make_tuple(
                    out_index.template at<Axis - ShrinkedDims>(),
                    fixed_dim_v<ShrinkedDims>);
            }
        }();

        auto new_in_index = in_index.append(new_dim);
        if constexpr (Axis + 1 < InRank) {
            return operator()<Axis + 1, decltype(new_shrinked_dims)::value>(
                new_in_index, out_index);
        } else {
            return new_in_index;
        }
    }
};
} // namespace detail

template <class Index, class Shape>
constexpr auto reduced_index_by_shape(const Index &src_index,
                                      const Shape &shape) noexcept {
    return generate_shape<Shape::rank()>([&](auto axis) {
        return detail::reduced_index_by_shape_dim<axis>(src_index, shape);
    });
}

template <size_t InRank, Shape TOutIndex, FixedDimensions TReduceAxes>
constexpr auto reduce_source_index_template(
    const TOutIndex &out_index,
    [[maybe_unused]] const TReduceAxes &reduce_axes) noexcept {
    // Keep dims
    if constexpr (InRank == TOutIndex::rank()) {
        return out_index;
    } else {
        return detail::reduce_source_begin_index_impl<InRank, TReduceAxes,
                                                      TOutIndex>{}(
            fixed_shape_v<>, out_index);
    }
}

template <Shape TInShape, FixedDimensions TReduceAxes>
constexpr auto sub_reduce_source_shape(
    const TInShape &in_shape,
    [[maybe_unused]] const TReduceAxes &reduce_axes) noexcept {
    return generate_shape<TInShape::rank()>([&](auto axis) {
        return ntt::select<TReduceAxes{}.contains(axis)>(in_shape[axis],
                                                         dim_one);
    });
}
} // namespace nncase::ntt::shape_infer
