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
#include "primitive_ops.h"
#include "tensor_traits.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace nncase::ntt {
using dim_t = int64_t;

template <dim_t Value> struct fixed_dim : std::integral_constant<dim_t, Value> {
    constexpr operator dim_t() const noexcept { return Value; }
};

template <dim_t Value> inline constexpr fixed_dim<Value> fixed_dim_v{};

inline constexpr fixed_dim<0> dim_zero{};
inline constexpr fixed_dim<1> dim_one{};

template <dim_t Value>
struct is_fixed_dim_t<fixed_dim<Value>> : std::true_type {};

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
template <FixedDimension TDim> constexpr auto operator-(const TDim &) noexcept {
    return fixed_dim_v<-TDim::value>;
}

#define DEFINE_NTT_DIM_BINARY_OP(op)                                           \
    template <Dimension TLhs, Dimension TRhs>                                  \
        requires(FixedDimension<TLhs> || FixedDimension<TRhs>)                 \
    constexpr auto operator op([[maybe_unused]] const TLhs &lhs,               \
                               [[maybe_unused]] const TRhs &rhs) noexcept {    \
        if constexpr (FixedDimension<TLhs> && FixedDimension<TRhs>) {          \
            return fixed_dim_v<TLhs::value op TRhs::value>;                    \
        } else {                                                               \
            return dim_value(lhs) op dim_value(rhs);                           \
        }                                                                      \
    }

DEFINE_NTT_DIM_BINARY_OP(+)
DEFINE_NTT_DIM_BINARY_OP(-)
DEFINE_NTT_DIM_BINARY_OP(*)
DEFINE_NTT_DIM_BINARY_OP(/)
DEFINE_NTT_DIM_BINARY_OP(%)

#undef DEFINE_NTT_DIM_BINARY_OP

#define DEFINE_NTT_DIM_COMPARE_OP(op)                                          \
    template <Dimension TLhs, Dimension TRhs>                                  \
        requires(FixedDimension<TLhs> || FixedDimension<TRhs>)                 \
    constexpr auto operator op([[maybe_unused]] const TLhs &lhs,               \
                               [[maybe_unused]] const TRhs &rhs) noexcept {    \
        if constexpr (FixedDimension<TLhs> && FixedDimension<TRhs>) {          \
            return std::integral_constant<bool,                                \
                                          (TLhs::value op TRhs::value)>{};     \
        } else {                                                               \
            return dim_value(lhs) op dim_value(rhs);                           \
        }                                                                      \
    }

DEFINE_NTT_DIM_COMPARE_OP(==)
DEFINE_NTT_DIM_COMPARE_OP(!=)
DEFINE_NTT_DIM_COMPARE_OP(<)
DEFINE_NTT_DIM_COMPARE_OP(<=)
DEFINE_NTT_DIM_COMPARE_OP(>)
DEFINE_NTT_DIM_COMPARE_OP(>=)

#undef DEFINE_NTT_DIM_COMPARE_OP

template <Dimension TNum, Dimension TDenom>
    requires(FixedDimension<TNum> || FixedDimension<TDenom>)
constexpr auto ceil_div([[maybe_unused]] const TNum &num,
                        [[maybe_unused]] const TDenom &denom) noexcept {
    if constexpr (FixedDimension<TNum> && FixedDimension<TDenom>) {
        return fixed_dim_v<(TNum::value + TDenom::value - 1) / TDenom::value>;
    } else {
        return (dim_value(num) + dim_value(denom) - 1) / dim_value(denom);
    }
}

template <Dimension TDim, Dimension TLowerBound, Dimension TUpperBound>
constexpr auto clamp(const TDim &dim, const TLowerBound &lower_bound,
                     const TUpperBound &upper_bound) noexcept {
    if constexpr (FixedDimension<TDim> && FixedDimension<TLowerBound> &&
                  FixedDimension<TUpperBound>) {
        static_assert(TLowerBound::value <= TUpperBound::value,
                      "Lower bound must be less than or equal to upper bound");
        return fixed_dim_v<std::clamp(TDim::value, TLowerBound::value,
                                      TUpperBound::value)>;
    } else {
        return std::clamp(dim_value(dim), dim_value(lower_bound),
                          dim_value(upper_bound));
    }
}

template <Dimension TM, Dimension TN>
    requires(FixedDimension<TM> || FixedDimension<TN>)
constexpr auto lcm([[maybe_unused]] const TM &m,
                   [[maybe_unused]] const TN &n) noexcept {
    if constexpr (FixedDimension<TM> && FixedDimension<TN>) {
        return fixed_dim_v<std::lcm(TM::value, TN::value)>;
    } else {
        return std::lcm(dim_value(m), dim_value(n));
    }
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

namespace detail {
template <class Cond, class T, class F> struct dim_where_impl;

template <class Cond, Dimension T, Dimension F>
struct dim_where_impl<Cond, T, F> {
    constexpr dim_t operator()(const Cond &cond, const T &true_dim,
                               const F &false_dim) const noexcept {
        return cond ? dim_value(true_dim) : dim_value(false_dim);
    }
};

template <bool Value, class T, class F>
struct dim_where_impl<std::integral_constant<bool, Value>, T, F> {
    constexpr auto
    operator()(const std::integral_constant<bool, Value> &,
               [[maybe_unused]] const T &true_dim,
               [[maybe_unused]] const F &false_dim) const noexcept {
        if constexpr (Value) {
            return true_dim;
        } else {
            return false_dim;
        }
    }
};
} // namespace detail

template <class Cond, class T, class F>
constexpr auto where(const Cond &cond, const T &true_dim,
                     const F &false_dim) noexcept {
    detail::dim_where_impl<Cond, T, F> impl;
    return impl(cond, true_dim, false_dim);
}
} // namespace nncase::ntt
