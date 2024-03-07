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
#include <array>
#include <cstddef>

namespace nncase::ntt {
template <class T, size_t... Lanes> struct native_vector_type;
//  {
// using value_type = T;

// using shape = fixed_shape<Lanes...>;
// using strides = default_strides_type<shape>;

// T elements[shape::length()];

// constexpr size_t length_at(size_t i) { return shape::at(i); }

// constexpr size_t rank() { return shape::rank(); }

// constexpr T operator[](size_t index) const noexcept {
//     return elements[index];
// }

// constexpr T operator()(size_t i, size_t j) const noexcept {
//     return elements[i * Width + j];
// }
// };

#ifdef __ARM_NEON__
#include <nncase/ntt/kernels/arch/arm/vector_types.h>
#endif

#ifdef __AVX__
#include <nncase/ntt/kernels/arch/x86_64/vector_types.h>
#endif

template <class T, size_t... Lanes> struct vector {
    using element_type = T;
    using value_type = typename native_vector_type<T, Lanes...>::type;
    using shape_type = fixed_shape<Lanes...>;
    using strides_type = default_strides_type<shape_type>::type;

  private:
    alignas(sizeof(value_type)) value_type v_;

  public:
    vector() = default;

    vector(const value_type &vec) : v_(vec) {}

    constexpr operator value_type() const noexcept { return v_; }

    constexpr operator value_type &() noexcept { return v_; }

    constexpr const shape_type shape() const noexcept { return shape_type{}; }

    constexpr const shape_type strides() const noexcept {
        return strides_type{};
    }

    constexpr auto buffer() noexcept {
        return std::span(reinterpret_cast<element_type *>(&v_),
                         shape_type::length());
    }

    template <class... Indices>
    constexpr const element_type &
    operator()(Indices &&...index) const noexcept {
        if constexpr (sizeof...(index) == 1 &&
                      (!std::is_integral_v<Indices> && ...)) {
            return buffer()[linear_offset(index..., strides_type{})];
        } else {
            return this->operator()(
                ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
        }
    }

    template <class... Indices>
    constexpr element_type &operator()(Indices &&...index) noexcept {
        if constexpr (sizeof...(index) == 1 &&
                      (!std::is_integral_v<Indices> && ...)) {
            return buffer()[linear_offset(index..., strides_type{})];
        } else {
            return this->operator()(
                ranked_shape<sizeof...(index)>{static_cast<size_t>(index)...});
        }
    }
};

} // namespace nncase::ntt