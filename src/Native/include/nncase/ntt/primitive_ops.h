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
#include "tensor_traits.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nncase::ntt {
namespace ops {

/**
 * @defgroup Unary operation functors
 * @{
 */

template <class T> struct abs {
    T operator()(const T &v) const noexcept { return std::abs(v); }
};

template <class T> struct acos {
    T operator()(const T &v) const noexcept { return std::acos(v); }
};

template <class T> struct acosh { T operator()(const T &v) const noexcept; };

template <class T> struct asin {
    T operator()(const T &v) const noexcept { return std::asin(v); }
};

template <class T> struct asinh { T operator()(const T &v) const noexcept; };

template <class T> struct ceil {
    T operator()(const T &v) const noexcept { return std::ceil(v); }
};

template <class T> struct cos {
    T operator()(const T &v) const noexcept { return std::cos(v); }
};

template <class T> struct cosh { T operator()(const T &v) const noexcept; };

template <class T> struct exp {
    T operator()(const T &v) const noexcept { return std::exp(v); }
};

template <class T> struct floor {
    T operator()(const T &v) const noexcept { return std::floor(v); }
};

template <class T> struct log {
    T operator()(const T &v) const noexcept { return std::log(v); }
};

template <class T> struct neg {
    T operator()(const T &v) const noexcept { return -v; }
};

template <class T> struct round {
    T operator()(const T &v) const noexcept { return std::nearbyint(v); }
};

template <class T> struct rsqrt {
    T operator()(const T &v) const noexcept { return (T)1 / std::sqrt(v); }
};

template <class T> struct sign {
    T operator()(const T &v) const noexcept {
        return (static_cast<T>(0) < v) - (v < static_cast<T>(0));
    }
};

template <class T> struct sin {
    T operator()(const T &v) const noexcept { return std::sin(v); }
};

template <class T> struct sinh { T operator()(const T &v) const noexcept; };

template <class T> struct sqrt {
    T operator()(const T &v) const noexcept { return std::sqrt(v); }
};

template <class T> struct square {
    T operator()(const T &v) const noexcept { return v * v; }
};

template <class T> struct tanh {
    T operator()(const T &v) const noexcept { return std::tanh(v); }
};

template <class T> struct swish { T operator()(const T &v) const noexcept; };

/**@}*/

/**
 * @defgroup Binary operation functors
 * @{
 */

template <class T1, class T2> struct add {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 + v2;
    }
};

template <class T1, class T2> struct sub {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 - v2;
    }
};

template <class T1, class T2> struct mul {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 * v2;
    }
};

template <class T1, class T2> struct div {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 / v2;
    }
};

/**
 * @remarks floor_mod is equivalent to % or mod() or remainder() function in
 * Python.
 */
template <class T1, class T2> struct floor_mod {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 -
               std::floor(static_cast<double>(v1) / static_cast<double>(v2)) *
                   v2;
    }
};

template <class T1, class T2> struct inner_product {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return v1 * v2;
    }
};

/**
 * @remarks mod is equivalent to fmod() function in C/C++/Python.
 */
template <class T1, class T2> struct mod {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return std::fmod(v1, v2);
    }
};

template <class T1, class T2> struct min {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return std::min(v1, v2);
    }
};

template <class T1, class T2> struct max {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return std::max(v1, v2);
    }
};

template <class T1, class T2> struct pow {
    auto operator()(const T1 &v1, const T2 &v2) const noexcept {
        return std::pow(v1, v2);
    }
};

/**@}*/

template <template <class T1, class T2> class BinaryOp, class TResult, class T>
struct reduce {
    TResult operator()(const T &v) const noexcept { return TResult(v); }
};

template <class T1, class T2, class TResult> struct mul_add {
    TResult operator()(const T1 &v1, const T2 &v2,
                       const TResult &v3) const noexcept;
};
} // namespace ops

#define NTT_DEFINE_UNARY_FUNC_IMPL(op)                                         \
    template <IsTensorOrScalar T> constexpr T op(const T &v) noexcept {        \
        return ops::op<T>()(v);                                                \
    }
#define NTT_DEFINE_BINARY_FUNC_IMPL(op)                                        \
    template <IsTensorOrScalar T1, IsTensorOrScalar T2>                        \
    constexpr auto op(const T1 &v1, const T2 &v2) noexcept {                   \
        return ops::op<T1, T2>()(v1, v2);                                      \
    }
#define NTT_DEFINE_REDUCE_FUNC_IMPL(name, op)                                  \
    template <class TResultOrVoid = void, IsTensorOrScalar T>                  \
    constexpr auto name(const T &v) noexcept {                                 \
        using TResult =                                                        \
            std::conditional_t<std::is_same_v<TResultOrVoid, void>,            \
                               element_or_scalar_t<T>, TResultOrVoid>;         \
        return ops::reduce<op, TResult, T>()(v);                               \
    }

NTT_DEFINE_UNARY_FUNC_IMPL(abs)
NTT_DEFINE_UNARY_FUNC_IMPL(acos)
NTT_DEFINE_UNARY_FUNC_IMPL(acosh)
NTT_DEFINE_UNARY_FUNC_IMPL(asin)
NTT_DEFINE_UNARY_FUNC_IMPL(asinh)
NTT_DEFINE_UNARY_FUNC_IMPL(ceil)
NTT_DEFINE_UNARY_FUNC_IMPL(cos)
NTT_DEFINE_UNARY_FUNC_IMPL(cosh)
NTT_DEFINE_UNARY_FUNC_IMPL(exp)
NTT_DEFINE_UNARY_FUNC_IMPL(floor)
NTT_DEFINE_UNARY_FUNC_IMPL(log)
NTT_DEFINE_UNARY_FUNC_IMPL(neg)
NTT_DEFINE_UNARY_FUNC_IMPL(round)
NTT_DEFINE_UNARY_FUNC_IMPL(rsqrt)
NTT_DEFINE_UNARY_FUNC_IMPL(sign)
NTT_DEFINE_UNARY_FUNC_IMPL(sin)
NTT_DEFINE_UNARY_FUNC_IMPL(sinh)
NTT_DEFINE_UNARY_FUNC_IMPL(sqrt)
NTT_DEFINE_UNARY_FUNC_IMPL(square)
NTT_DEFINE_UNARY_FUNC_IMPL(tanh)
NTT_DEFINE_UNARY_FUNC_IMPL(swish)

NTT_DEFINE_BINARY_FUNC_IMPL(add)
NTT_DEFINE_BINARY_FUNC_IMPL(sub)
NTT_DEFINE_BINARY_FUNC_IMPL(mul)
NTT_DEFINE_BINARY_FUNC_IMPL(div)
NTT_DEFINE_BINARY_FUNC_IMPL(floor_mod)
NTT_DEFINE_BINARY_FUNC_IMPL(inner_product)
NTT_DEFINE_BINARY_FUNC_IMPL(mod)
NTT_DEFINE_BINARY_FUNC_IMPL(min)
NTT_DEFINE_BINARY_FUNC_IMPL(max)
NTT_DEFINE_BINARY_FUNC_IMPL(pow)

NTT_DEFINE_REDUCE_FUNC_IMPL(reduce_sum, ops::add)
NTT_DEFINE_REDUCE_FUNC_IMPL(reduce_max, ops::max)

template <IsTensorOrScalar T1, IsTensorOrScalar T2, IsTensorOrScalar TResult>
constexpr TResult mul_add(const T1 &v1, const T2 &v2,
                          const TResult &v3) noexcept {
    return ops::mul_add<T1, T2, TResult>()(v1, v2, v3);
}

/**
 * @defgroup Builtin operators
 * @{
 */

template <IsTensorOrScalar T> constexpr T operator-(const T &value) noexcept {
    return neg(value);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr auto operator+(const T1 &v1, const T2 &v2) noexcept {
    return add(v1, v2);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr auto operator-(const T1 &v1, const T2 &v2) noexcept {
    return sub(v1, v2);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr auto operator*(const T1 &v1, const T2 &v2) noexcept {
    return mul(v1, v2);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr auto operator/(const T1 &v1, const T2 &v2) noexcept {
    return div(v1, v2);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr auto operator%(const T1 &v1, const T2 &v2) noexcept {
    return mod(v1, v2);
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr T1 &operator+=(T1 &v1, const T2 &v2) noexcept {
    v1 = add(v1, v2);
    return v1;
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr T1 &operator-=(T1 &v1, const T2 &v2) noexcept {
    v1 = sub(v1, v2);
    return v1;
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr T1 &operator*=(T1 &v1, const T2 &v2) noexcept {
    v1 = mul(v1, v2);
    return v1;
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr T1 &operator/=(T1 &v1, const T2 &v2) noexcept {
    v1 = div(v1, v2);
    return v1;
}

template <IsTensorOrScalar T1, IsTensorOrScalar T2>
constexpr T1 &operator%=(T1 &v1, const T2 &v2) noexcept {
    v1 = mod(v1, v2);
    return v1;
}

/**@}*/

// complex ops

namespace ops {
// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
template <class T> T acosh<T>::operator()(const T &v) const noexcept {
    return ntt::log(v + ntt::sqrt(v * v - 1));
}

// asinh(v) = ln(v + sqrt(v^2 + 1))
template <class T> T asinh<T>::operator()(const T &v) const noexcept {
    return ntt::log(v + ntt::sqrt(v * v + 1));
}

// cosh(v) = (exp(v) + exp(-v)) / 2
template <class T> T cosh<T>::operator()(const T &v) const noexcept {
    return (ntt::exp(v) + ntt::exp(-v)) / 2;
}

// sinh(v) = (exp(v) - exp(-v)) / 2
template <class T> T sinh<T>::operator()(const T &v) const noexcept {
    return (ntt::exp(v) - ntt::exp(-v)) / 2;
}

// swish(v) = v / (exp(-v) + 1)
template <class T> T swish<T>::operator()(const T &v) const noexcept {
    return v / (ntt::exp(-v) + 1);
}

template <class T1, class T2, class TResult>
TResult mul_add<T1, T2, TResult>::operator()(const T1 &v1, const T2 &v2,
                                             const TResult &v3) const noexcept {
    return v1 * v2 + v3;
}
} // namespace ops
} // namespace nncase::ntt
