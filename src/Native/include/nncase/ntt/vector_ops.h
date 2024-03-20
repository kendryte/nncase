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
#include "vector.h"

namespace nncase::ntt::ops {
// unary_ops ops
namespace detail {
template <template <class T> class Op, class TVec> struct vector_unary_impl {
    using element_type = typename TVec::element_type;

    TVec operator()(TVec v) const noexcept {
        Op<element_type> op;
        for (auto &elem : v.elements()) {
            elem = op(elem);
        }
        return v;
    }
};
} // namespace detail

#define NTT_DEFINE_VECTOR_UNARY_IMPL(op)                                       \
    template <class T, size_t... Lanes>                                        \
    struct op<vector<T, Lanes...>>                                             \
        : detail::vector_unary_impl<op, vector<T, Lanes...>> {}

NTT_DEFINE_VECTOR_UNARY_IMPL(abs);
NTT_DEFINE_VECTOR_UNARY_IMPL(acos);
NTT_DEFINE_VECTOR_UNARY_IMPL(acosh);
NTT_DEFINE_VECTOR_UNARY_IMPL(asin);
NTT_DEFINE_VECTOR_UNARY_IMPL(asinh);
NTT_DEFINE_VECTOR_UNARY_IMPL(ceil);
NTT_DEFINE_VECTOR_UNARY_IMPL(cos);
NTT_DEFINE_VECTOR_UNARY_IMPL(cosh);
NTT_DEFINE_VECTOR_UNARY_IMPL(exp);
NTT_DEFINE_VECTOR_UNARY_IMPL(floor);
NTT_DEFINE_VECTOR_UNARY_IMPL(log);
NTT_DEFINE_VECTOR_UNARY_IMPL(neg);
NTT_DEFINE_VECTOR_UNARY_IMPL(round);
NTT_DEFINE_VECTOR_UNARY_IMPL(rsqrt);
NTT_DEFINE_VECTOR_UNARY_IMPL(sign);
NTT_DEFINE_VECTOR_UNARY_IMPL(sin);
NTT_DEFINE_VECTOR_UNARY_IMPL(sinh);
NTT_DEFINE_VECTOR_UNARY_IMPL(sqrt);
NTT_DEFINE_VECTOR_UNARY_IMPL(square);
NTT_DEFINE_VECTOR_UNARY_IMPL(tanh);
NTT_DEFINE_VECTOR_UNARY_IMPL(swish);
} // namespace nncase::ntt::ops

namespace nncase::ntt::vector_ops {
template <class TVec> struct load {
    using T = typename TVec::element_type;

    TVec operator()(const T *src) const noexcept {
        TVec vec;
        std::copy(src, src + vec.size(), vec.elements().data());
        return vec;
    }
};

template <class TVec> struct load_scalar {
    using T = typename TVec::element_type;

    TVec operator()(T value) const noexcept {
        TVec vec;
        std::fill_n(vec.elements().data(), vec.size(), value);
        return vec;
    }
};

template <class TVec> struct reduce_sum;
template <class TVec> struct reduce_max;
} // namespace nncase::ntt::vector_ops

namespace nncase::ntt {
template <class T, size_t... Lanes>
vector<T, Lanes...> vector<T, Lanes...>::from_scalar(T v) noexcept {
    return vector_ops::load_scalar<vector<T, Lanes...>>()(v);
}
} // namespace nncase::ntt
