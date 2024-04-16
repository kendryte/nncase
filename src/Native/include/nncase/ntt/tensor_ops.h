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
#include "tensor.h"
#include "tensor_traits.h"

namespace nncase::ntt::ops {
// unary_ops ops
namespace detail {
template <template <class T> class Op, IsTensor TTensor>
struct tensor_unary_impl {
    using element_type = typename TTensor::element_type;

    TTensor operator()(const TTensor &v) const noexcept {
        TTensor value;
        apply(v.shape(), [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

  private:
    Op<element_type> op_;
};

template <template <class T1, class T2> class Op, IsTensor TTensor, class T2>
struct tensor_binary_impl {
    using element_type1 = typename TTensor::element_type;
    using element_type2 = element_or_scalar_t<T2>;

    TTensor operator()(const TTensor &v1, const T2 &v2) const noexcept {
        TTensor value;
        if constexpr (IsTensor<T2>) {
            apply(v1.shape(), [&](auto index) {
                value(index) = op_(v1(index), v2(index));
            });
        } else {
            apply(v1.shape(),
                  [&](auto index) { value(index) = op_(v1(index), v2); });
        }

        return value;
    }

  private:
    Op<element_type1, element_type2> op_;
};
} // namespace detail

#define NTT_DEFINE_TENSOR_UNARY_IMPL(op)                                       \
    template <IsTensor TTensor>                                                \
    struct op<TTensor> : detail::tensor_unary_impl<op, TTensor> {}

#define NTT_DEFINE_TENSOR_BINARY_IMPL(op)                                      \
    template <IsTensor TTensor, class T2>                                      \
    struct op<TTensor, T2> : detail::tensor_binary_impl<op, TTensor, T2> {}

NTT_DEFINE_TENSOR_UNARY_IMPL(abs);
NTT_DEFINE_TENSOR_UNARY_IMPL(acos);
NTT_DEFINE_TENSOR_UNARY_IMPL(asin);
NTT_DEFINE_TENSOR_UNARY_IMPL(ceil);
NTT_DEFINE_TENSOR_UNARY_IMPL(cos);
NTT_DEFINE_TENSOR_UNARY_IMPL(exp);
NTT_DEFINE_TENSOR_UNARY_IMPL(floor);
NTT_DEFINE_TENSOR_UNARY_IMPL(log);
NTT_DEFINE_TENSOR_UNARY_IMPL(neg);
NTT_DEFINE_TENSOR_UNARY_IMPL(round);
NTT_DEFINE_TENSOR_UNARY_IMPL(rsqrt);
NTT_DEFINE_TENSOR_UNARY_IMPL(sign);
NTT_DEFINE_TENSOR_UNARY_IMPL(sin);
NTT_DEFINE_TENSOR_UNARY_IMPL(sqrt);
NTT_DEFINE_TENSOR_UNARY_IMPL(tanh);

NTT_DEFINE_TENSOR_BINARY_IMPL(add);
NTT_DEFINE_TENSOR_BINARY_IMPL(sub);
NTT_DEFINE_TENSOR_BINARY_IMPL(mul);
NTT_DEFINE_TENSOR_BINARY_IMPL(div);
NTT_DEFINE_TENSOR_BINARY_IMPL(floor_mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(min);
NTT_DEFINE_TENSOR_BINARY_IMPL(max);
NTT_DEFINE_TENSOR_BINARY_IMPL(pow);

template <template <class T1, class T2> class Op, class TResult,
          IsTensor TTensor>
struct reduce<Op, TResult, TTensor> {
    using element_type = typename TTensor::element_type;

    TResult operator()(const TTensor &v) const noexcept {
        Op<TResult, element_type> op;
        auto elements = v.elements();
        auto it = elements.begin();
        auto value = TResult(*it++);
        for (; it != elements.end(); ++it) {
            value = op(value, *it);
        }
        return value;
    }
};
} // namespace nncase::ntt::ops

namespace nncase::ntt::tensor_ops {

template <class TTensor> struct load {
    using T = typename TTensor::element_type;

    TTensor operator()(const T *src) const noexcept {
        TTensor vec;
        std::copy(src, src + vec.size(), vec.elements().data());
        return vec;
    }
};

template <class TTensor> struct load_scalar {
    using T = typename TTensor::element_type;

    TTensor operator()(const T &value) const noexcept {
        TTensor vec;
        std::fill_n(vec.elements().data(), vec.size(), value);
        return vec;
    }
};
} // namespace nncase::ntt::tensor_ops

namespace nncase::ntt {
template <class T, class Shape, class Strides, size_t MaxSize>
detail::tensor_impl<T, Shape, Strides, MaxSize, false, true>::tensor_impl(
    T value) noexcept
    : tensor_impl(tensor_ops::load_scalar<
                  tensor_base<T, Shape, Strides, MaxSize, false>>()(value)) {}

template <class T, class Shape, class Strides, size_t MaxSize, bool IsView>
tensor_base<T, Shape, Strides, MaxSize, IsView>
tensor_base<T, Shape, Strides, MaxSize, IsView>::from_scalar(T v) {
    return tensor_ops::load_scalar<
        tensor_base<T, Shape, Strides, MaxSize, IsView>>()(v);
}
} // namespace nncase::ntt
