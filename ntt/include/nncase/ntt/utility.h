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
#include "shape.h"
#include "tensor_traits.h"
#include <cstring>
#include <type_traits>

namespace nncase::ntt::utility_detail {
template <size_t Axis, Tensor TTensor, Shape TOutShape>
constexpr auto get_safe_stride(const TTensor &tensor,
                               const TOutShape &out_shape) noexcept {
    auto dim_ext = out_shape.rank() - tensor.rank();
    if constexpr (Axis < dim_ext) {
        return dim_zero;
    } else {
        auto actual_axis = fixed_dim_v<Axis> - dim_ext;
        auto actual_dim = tensor.shape()[actual_axis];
        if constexpr (FixedDimension<std::decay_t<decltype(actual_dim)>>) {
            if constexpr (actual_dim == 1) {
                return dim_zero;
            } else {
                return tensor.strides()[actual_axis];
            }
        } else {
            if (actual_dim == 1) {
                return (dim_t)0;
            } else {
                return dim_value(tensor.strides()[actual_axis]);
            }
        }
    }
}
} // namespace nncase::ntt::utility_detail
