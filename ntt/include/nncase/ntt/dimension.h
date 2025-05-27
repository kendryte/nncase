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
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
using dim_t = int64_t;

template <dim_t Value> struct fixed_dim : std::integral_constant<dim_t, Value> {
    constexpr operator dim_t() const noexcept { return Value; }
};

template <dim_t Value> inline constexpr fixed_dim<Value> fixed_dim_v{};

inline constexpr fixed_dim<0> dim_zero{};
inline constexpr fixed_dim<1> dim_one{};

template <class T> struct is_fixed_dim_t : std::false_type {};

template <dim_t Value>
struct is_fixed_dim_t<fixed_dim<Value>> : std::true_type {};

template <class T>
inline constexpr bool is_fixed_dim_v = is_fixed_dim_t<T>::value;

template <class T>
concept DynamicDimension = std::is_integral_v<T>;

template <class T>
concept Dimension = is_fixed_dim_v<T> || DynamicDimension<T>;

template <class T>
concept FixedDimension = is_fixed_dim_v<T>;

template <Dimension T> constexpr auto dim_value(const T &dim) noexcept {
    if constexpr (is_fixed_dim_v<std::decay_t<T>>) {
        return T::value;
    } else {
        return dim;
    }
}

namespace detail {
template <typename t> constexpr t pow(t base, int exp) {
    return (exp > 0) ? base * pow(base, exp - 1) : 1;
};

template <char...> struct char_literal;
template <> struct char_literal<> {
    static const dim_t to_int = 0;
};
template <char c, char... cv> struct char_literal<c, cv...> {
    static const dim_t to_int = c == '-' ? -1 * char_literal<cv...>::to_int
                                         : (c - '0') * pow(10, sizeof...(cv)) +
                                               char_literal<cv...>::to_int;
};
} // namespace detail

template <char... cv> inline constexpr auto operator"" _dim() {
    constexpr auto value = detail::char_literal<cv...>::to_int;
    return fixed_dim_v<value>;
}

template <FixedDimension TLhs, FixedDimension TRhs>
constexpr auto operator+(const TLhs &, const TRhs &) noexcept {
    return fixed_dim_v<TLhs::value + TRhs::value>;
}

template <FixedDimension TLhs, FixedDimension TRhs>
constexpr auto operator-(const TLhs &, const TRhs &) noexcept {
    return fixed_dim_v<TLhs::value - TRhs::value>;
}

template <FixedDimension TLhs, FixedDimension TRhs>
constexpr auto operator*(const TLhs &, const TRhs &) noexcept {
    return fixed_dim_v<TLhs::value * TRhs::value>;
}

template <FixedDimension TLhs, FixedDimension TRhs>
constexpr auto operator/(const TLhs &, const TRhs &) noexcept {
    return fixed_dim_v<TLhs::value / TRhs::value>;
}

template <FixedDimension TLhs, FixedDimension TRhs>
constexpr auto operator%(const TLhs &, const TRhs &) noexcept {
    return fixed_dim_v<TLhs::value % TRhs::value>;
}

template <Dimension... TDims>
constexpr auto min(const TDims &...dims) noexcept {
    if constexpr ((... && FixedDimension<TDims>)) {
        return fixed_dim_v<std::min({TDims::value...})>;
    } else {
        return std::min({dim_value(dims)...});
    }
}

template <Dimension... TDims>
constexpr auto max(const TDims &...dims) noexcept {
    if constexpr ((... && FixedDimension<TDims>)) {
        return fixed_dim_v<std::max({TDims::value...})>;
    } else {
        return std::max({dim_value(dims)...});
    }
}

template <Dimension TIndex, Dimension TExtent>
constexpr auto positive_index(const TIndex &index,
                              [[maybe_unused]] const TExtent &dim) {
    if constexpr (FixedDimension<TIndex>) {
        if constexpr (TIndex::value < 0) {
            return index + dim;
        } else {
            return index;
        }
    } else {
        return index < 0 ? index + dim : index;
    }
}

template <bool Cond, class T, class F>
constexpr decltype(auto) select(T &&true_value, F &&false_value) {
    if constexpr (Cond) {
        return std::forward<T>(true_value);
    } else {
        return std::forward<F>(false_value);
    }
}
} // namespace nncase::ntt
