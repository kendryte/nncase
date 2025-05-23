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
#include <cstring>

namespace nncase::ntt {
template <typename T>
concept Vector = std::decay_t<T>::IsVector;

template <typename T>
concept ShardedTensor = requires {
    typename T::sharding_type;
    typename T::mesh_type;
};

template <typename T>
concept Tensor = requires {
    typename T::shape_type;
    typename T::strides_type;
} && !ShardedTensor<T> && !Vector<T>;

template <typename T>
concept FixedTensor = Tensor<T> && FixedDimensions<typename T::shape_type> &&
                      FixedDimensions<typename T::strides_type>;

template <typename T>
concept Scalar = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
concept ScalarOrVector = Scalar<T> || Vector<T>;

template <typename T>
concept TensorOrScalar = Tensor<T> || Scalar<T>;

template <class T> struct element_or_scalar_type {
    using type = T;
};

template <Tensor T> struct element_or_scalar_type<T> {
    using type = typename T::element_type;
};

template <Vector T> struct element_or_scalar_type<T> {
    using type = typename T::element_type;
};

template <class T>
using element_or_scalar_t = typename element_or_scalar_type<T>::type;
} // namespace nncase::ntt
