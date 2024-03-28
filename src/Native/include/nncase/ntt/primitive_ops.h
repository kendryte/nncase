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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nncase::ntt {
namespace ops {
// unary_ops ops

template <class T> struct abs {
    T operator()(T v) const noexcept { return std::abs(v); }
};

template <class T> struct acos {
    T operator()(T v) const noexcept { return std::acos(v); }
};

template <class T> struct acosh {
    T operator()(T v) const noexcept { return std::acosh(v); }
};

template <class T> struct asin {
    T operator()(T v) const noexcept { return std::asin(v); }
};

template <class T> struct asinh {
    T operator()(T v) const noexcept { return std::asinh(v); }
};

template <class T> struct ceil {
    T operator()(T v) const noexcept { return std::ceil(v); }
};

template <class T> struct cos {
    T operator()(T v) const noexcept { return std::cos(v); }
};

template <class T> struct cosh {
    T operator()(T v) const noexcept { return std::cosh(v); }
};

template <class T> struct exp {
    T operator()(T v) const noexcept { return std::exp(v); }
};

template <class T> struct floor {
    T operator()(T v) const noexcept { return std::floor(v); }
};

template <class T> struct log {
    T operator()(T v) const noexcept { return std::log(v); }
};

template <class T> struct neg {
    T operator()(T v) const noexcept { return -v; }
};

template <class T> struct round {
    T operator()(T v) const noexcept { return std::nearbyint(v); }
};

template <class T> struct rsqrt {
    T operator()(T v) const noexcept { return (T)1 / std::sqrt(v); }
};

template <class T> struct sign {
    T operator()(T v) const noexcept { return (static_cast<T>(0) < v) - (v < static_cast<T>(0)); }
};

template <class T> struct sin {
    T operator()(T v) const noexcept { return std::sin(v); }
};

template <class T> struct sinh {
    T operator()(T v) const noexcept { return std::sinh(v); }
};

template <class T> struct sqrt {
    T operator()(T v) const noexcept { return std::sqrt(v); }
};

template <class T> struct square {
    T operator()(T v) const noexcept { return v * v; }
};

template <class T> struct tanh {
    T operator()(T v) const noexcept { return std::tanh(v); }
};

template <class T> struct swish {
    T operator()(T v) const noexcept { return v / (1 + std::exp(-v)); }
};

// binary ops

template <class T> struct add {
    T operator()(T v1, T v2) const noexcept { return v1 + v2; }
};

template <class T> struct sub {
    T operator()(T v1, T v2) const noexcept { return v1 - v2; }
};

template <class T> struct mul {
    T operator()(T v1, T v2) const noexcept { return v1 * v2; }
};

template <class T> struct div {
    T operator()(T v1, T v2) const noexcept { return v1 / v2; }
};

// floor_mod is equivalent to % or mod() or remainder() function in Python.
template <class T> struct floor_mod {
    T operator()(T v1, T v2) const noexcept {
        return v1 - std::floor(static_cast<double>(v1) / static_cast<double>(v2)) * v2;
    }
};

// mod is equivalent to fmod() function in C/C++/Python.
template <class T> struct mod {
    T operator()(T v1, T v2) const noexcept {
        return std::fmod(v1, v2);
    }
};

template <class T> struct min {
    T operator()(T v1, T v2) const noexcept { return std::min(v1, v2); }
};

template <class T> struct max {
    T operator()(T v1, T v2) const noexcept { return std::max(v1, v2); }
};

template <class T> struct pow {
    T operator()(T v1, T v2) const noexcept { return std::pow(v1, v2); }
};
} // namespace ops

#define NTT_DEFINE_UNARY_FUNC_IMPL(op)                                         \
    template <class T> constexpr T op(T value) noexcept {                      \
        return ops::op<T>()(value);                                            \
    }
#define NTT_DEFINE_BINARY_FUNC_IMPL(op)                                        \
    template <class T> constexpr T op(T v1, T v2) noexcept {                   \
        return ops::op<T>()(v1, v2);                                           \
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
NTT_DEFINE_BINARY_FUNC_IMPL(mod)
NTT_DEFINE_BINARY_FUNC_IMPL(min)
NTT_DEFINE_BINARY_FUNC_IMPL(max)
NTT_DEFINE_BINARY_FUNC_IMPL(pow)

// operators

template <class T> constexpr T operator-(T value) noexcept {
    return neg(value);
}

template <class T> constexpr T operator+(T v1, T v2) noexcept {
    return add(v1, v2);
}

template <class T> constexpr T operator-(T v1, T v2) noexcept {
    return sub(v1, v2);
}

template <class T> constexpr T operator*(T v1, T v2) noexcept {
    return mul(v1, v2);
}

template <class T> constexpr T operator/(T v1, T v2) noexcept {
    return div(v1, v2);
}

template <class T> constexpr T operator%(T v1, T v2) noexcept {
    return mod(v1, v2);
}
} // namespace nncase::ntt
