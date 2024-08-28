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
#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

namespace nncase::ntt {
template <typename T>
concept IsFixedTensor = is_fixed_dims_v<typename std::decay_t<T>::shape_type>
    &&is_fixed_dims_v<typename std::decay_t<T>::strides_type>;

template <typename T>
concept IsRankedTensor = is_ranked_dims_v<typename std::decay_t<T>::shape_type>
    &&is_ranked_dims_v<typename std::decay_t<T>::strides_type>;

template <typename T> concept IsVector = std::decay_t<T>::IsVector;

template <typename T>
concept IsScalar = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T> concept IsTensor = IsFixedTensor<T> || IsRankedTensor<T>;

template <typename T> concept IsTensorOrScalar = IsTensor<T> || IsScalar<T>;

template <typename T> concept IsFixedDims = is_fixed_dims_v<T>;

template <class T> struct element_or_scalar_type { using type = T; };

template <IsTensor T> struct element_or_scalar_type<T> {
    using type = typename T::element_type;
};

template <class T>
using element_or_scalar_t = typename element_or_scalar_type<T>::type;

template <class T, size_t... Lanes> struct fixed_tensor_alike_type;
template <class T, size_t... Lanes>
using fixed_tensor_alike_t =
    typename fixed_tensor_alike_type<T, Lanes...>::type;
} // namespace nncase::ntt
