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
#include <cstring>
#include <type_traits>

namespace nncase::ntt {
enum dims_usage {
    normal,
    shape,
    strides,
};

template <class T> struct is_fixed_dim_t : std::false_type {};

template <class T>
inline constexpr bool is_fixed_dim_v =
    is_fixed_dim_t<std::remove_cv_t<T>>::value;

template <class T>
concept DynamicDimension = std::is_integral_v<T>;

template <class T>
concept Dimension = is_fixed_dim_v<T> || DynamicDimension<T>;

template <class T>
concept FixedDimension = is_fixed_dim_v<T>;

template <class T>
concept Dimensions = requires {
    T::rank();
    T::fixed_rank();
    T::dynamic_rank();
    T::usage();
};

template <class T>
concept FixedDimensions = Dimensions<T> && T::is_fixed();

template <class T>
concept Shape = Dimensions<T> && T::usage() == dims_usage::shape;

template <class T>
concept FixedShape = Shape<T> && T::is_fixed();

template <class T>
concept Strides = Dimensions<T> && T::usage() == dims_usage::strides;

template <class T>
concept FixedStrides = Strides<T> && T::is_fixed();

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

template <typename T>
concept TensorOrVector = Tensor<T> || Vector<T>;

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

template <class T>
struct element_scalar_count : std::integral_constant<size_t, 1> {};

template <Vector T>
struct element_scalar_count<T>
    : std::integral_constant<size_t, std::remove_cv_t<T>::size()> {};

template <class T>
inline constexpr size_t element_scalar_count_v = element_scalar_count<T>::value;

template <class T, class U> struct replace_element_type {
    using type = U;
};

template <class T, class U>
using replace_element_t = typename replace_element_type<T, U>::type;
} // namespace nncase::ntt
