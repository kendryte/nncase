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

namespace nncase::ntt {
// math ops
namespace mathops {
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

template <class T> struct mod {
    T operator()(T v1, T v2) const noexcept { return v1 % v2; }
};

template <class T> struct min {
    T operator()(T v1, T v2) const noexcept { return std::min(v1, v2); }
};

template <class T> struct max {
    T operator()(T v1, T v2) const noexcept { return std::max(v1, v2); }
};
} // namespace mathops

template <template <class T> class Op, class TLhs, class TRhs, class TOut>
void binary(const TLhs &lhs, const TRhs &rhs, TOut &&output) {
    Op<typename TLhs::element_type> op;
    for (size_t i = 0; i < lhs.buffer().size(); i++) {
        output.buffer()[i] = op(lhs.buffer()[i]);
    }
}
} // namespace nncase::ntt
