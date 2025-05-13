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
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nncase::ntt {
using dim_t = int64_t;

template <dim_t Value> struct fixed_dim : std::integral_constant<dim_t, Value> {
    constexpr operator dim_t() const noexcept { return Value; }
};

template <dim_t Value> inline constexpr fixed_dim<Value> fixed_dim_v{};

inline constexpr fixed_dim<0> fixed_dim_zero{};
inline constexpr fixed_dim<1> fixed_dim_one{};

template <class T> struct is_fixed_dim_t : std::false_type {};

template <dim_t Value>
struct is_fixed_dim_t<fixed_dim<Value>> : std::true_type {};

template <class T>
inline constexpr bool is_fixed_dim_v = is_fixed_dim_t<T>::value;

template <class T>
concept Dimension = is_fixed_dim_v<T> || std::is_integral_v<T>;

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
} // namespace nncase::ntt
