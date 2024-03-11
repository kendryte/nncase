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
#include <type_traits>

namespace nncase::ntt {
template <class T, size_t... Lanes> struct native_vector_type;
}

#ifdef __ARM_NEON__
#include <nncase/ntt/kernels/arch/arm/vector_types.h>
#endif

#ifdef __AVX__
#include <nncase/ntt/kernels/arch/x86_64/vector_types.h>
#endif

namespace nncase::ntt {

template <class T, size_t... Lanes> struct vector {
    using element_type = T;
    using value_type = native_vector_type<T, Lanes...>;
    using native_type = typename value_type::type;
    using shape_type = fixed_shape<Lanes...>;
    using strides_type = typename default_strides_type<shape_type>::type;

  private:
    alignas(sizeof(native_type)) native_type v_;

  public:
    vector() = default;

    vector(const native_type &vec) : v_(vec) {}

    vector(const element_type &v) : v_(value_type::from_element(v)) {}

    template <typename TOther>
        requires std::is_nothrow_convertible_v<TOther, element_type>
    vector(const TOther &v) : v_(value_type::from_element(v)) {}

    constexpr operator native_type() const noexcept { return v_; }

    constexpr operator native_type &() noexcept { return v_; }

    static constexpr auto shape() noexcept { return shape_type{}; }

    static constexpr auto strides() noexcept { return strides_type{}; }

    // NOTE we can't assume the simd type have the compiler support.
    // constexpr vector<T, Lanes...> operator+(const vector<T, Lanes...> &rhs) {
    //     return v_ + rhs.v_;
    // }

    // constexpr vector<T, Lanes...> operator-(const vector<T, Lanes...> &rhs) {
    //     return v_ - rhs.v_;
    // }

    // constexpr vector<T, Lanes...> operator*(const vector<T, Lanes...> &rhs) {
    //     return v_ * rhs.v_;
    // }

    // constexpr vector<T, Lanes...> operator/(const vector<T, Lanes...> &rhs) {
    //     return v_ / rhs.v_;
    // }

    constexpr auto buffer() noexcept {
        return std::span(reinterpret_cast<element_type *>(&v_),
                         shape_type::length());
    }

    constexpr auto buffer() const noexcept {
        return std::span(reinterpret_cast<const element_type *>(&v_),
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