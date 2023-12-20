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
#include "../shape_infer/binary.h"
#include "../shape_infer/reduce.h"
#include <type_traits>

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
    T operator()(T v1, T v2) const noexcept {
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            return std::fmod(v1, v2);
        } else {
            return v1 % v2;
        }
    }
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
    auto out_shape = shape_infer::binary_output_shape(lhs.shape(), rhs.shape());

    apply(out_shape, [&](auto index) {
        const auto lhs_index =
            shape_infer::reduced_index_by_shape(index, lhs.shape());
        const auto rhs_index =
            shape_infer::reduced_index_by_shape(index, rhs.shape());
        output(index) = op(lhs(lhs_index), rhs(rhs_index));
    });
}
} // namespace nncase::ntt
