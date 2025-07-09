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
template <size_t Axis, Shape TShape>
constexpr auto sub_matmul_shape_dim(const TShape &shape) noexcept {
    if constexpr (Axis >= TShape::rank() - 2) {
        return shape.template at<Axis>();
    } else {
        return dim_one;
    }
}
} // namespace detail

template <Shape TShape> constexpr auto sub_matmul_shape(const TShape &shape) {
    static_assert(TShape::rank() >= 2,
                  "matmul shape must have at least 2 dimensions");
    return generate_shape<TShape::rank()>([&shape](auto axis) {
        return detail::sub_matmul_shape_dim<axis>(shape);
    });
}
} // namespace nncase::ntt::shape_infer
