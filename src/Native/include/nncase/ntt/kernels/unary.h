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
#include "../apply.h"

namespace nncase::ntt {
// math ops
namespace mathops {
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
    T operator()(T v) const noexcept { return std::copysign((T)1, v); }
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
} // namespace mathops

template <template <class T> class Op, class TA, class TB>
void unary(const TA &input, TB &&output) {
    Op<typename TA::element_type> op;
    apply(input.shape(), [&](auto index) { output(index) = op(input(index)); });
}
} // namespace nncase::ntt
