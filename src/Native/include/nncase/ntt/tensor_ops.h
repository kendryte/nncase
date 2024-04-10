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

namespace nncase::ntt::ops {
// unary_ops ops
namespace detail {
template <template <class T> class Op, class TTensor> struct tensor_unary_impl {
    using element_type = typename TTensor::element_type;

    TTensor operator()(const TTensor &v) const noexcept {
        Op<element_type> op;
        TTensor value;
        apply(v.shape(), [&](auto index) { value(index) = op(v(index)); });
        return value;
    }
};

template <template <class T> class Op, class TTensor>
struct tensor_binary_impl {
    using element_type = typename TTensor::element_type;

    TTensor operator()(const TTensor &v1, const TTensor &v2) const noexcept {
        Op<element_type> op;
        TTensor value;
        apply(v1.shape(),
              [&](auto index) { value(index) = op(v1(index), v2(index)); });
        return value;
    }
};
} // namespace detail

#define NTT_DEFINE_TENSOR_UNARY_IMPL(op)                                       \
    template <class T, class Shape, class Strides, size_t MaxSize,             \
              bool IsView>                                                     \
    struct op<tensor_base<T, Shape, Strides, MaxSize, IsView>>                 \
        : detail::tensor_unary_impl<                                           \
              op, tensor_base<T, Shape, Strides, MaxSize, IsView>> {}

#define NTT_DEFINE_TENSOR_BINARY_IMPL(op)                                      \
    template <class T, class Shape, class Strides, size_t MaxSize,             \
              bool IsView>                                                     \
    struct op<tensor_base<T, Shape, Strides, MaxSize, IsView>>                 \
        : detail::tensor_binary_impl<                                          \
              op, tensor_base<T, Shape, Strides, MaxSize, IsView>> {}

NTT_DEFINE_TENSOR_UNARY_IMPL(abs);
NTT_DEFINE_TENSOR_UNARY_IMPL(acos);
NTT_DEFINE_TENSOR_UNARY_IMPL(acosh);
NTT_DEFINE_TENSOR_UNARY_IMPL(asin);
NTT_DEFINE_TENSOR_UNARY_IMPL(asinh);
NTT_DEFINE_TENSOR_UNARY_IMPL(ceil);
NTT_DEFINE_TENSOR_UNARY_IMPL(cos);
NTT_DEFINE_TENSOR_UNARY_IMPL(cosh);
NTT_DEFINE_TENSOR_UNARY_IMPL(exp);
NTT_DEFINE_TENSOR_UNARY_IMPL(floor);
NTT_DEFINE_TENSOR_UNARY_IMPL(log);
NTT_DEFINE_TENSOR_UNARY_IMPL(neg);
NTT_DEFINE_TENSOR_UNARY_IMPL(round);
NTT_DEFINE_TENSOR_UNARY_IMPL(rsqrt);
NTT_DEFINE_TENSOR_UNARY_IMPL(sign);
NTT_DEFINE_TENSOR_UNARY_IMPL(sin);
NTT_DEFINE_TENSOR_UNARY_IMPL(sinh);
NTT_DEFINE_TENSOR_UNARY_IMPL(sqrt);
NTT_DEFINE_TENSOR_UNARY_IMPL(square);
NTT_DEFINE_TENSOR_UNARY_IMPL(tanh);
NTT_DEFINE_TENSOR_UNARY_IMPL(swish);

NTT_DEFINE_TENSOR_BINARY_IMPL(add);
NTT_DEFINE_TENSOR_BINARY_IMPL(sub);
NTT_DEFINE_TENSOR_BINARY_IMPL(mul);
NTT_DEFINE_TENSOR_BINARY_IMPL(div);
NTT_DEFINE_TENSOR_BINARY_IMPL(floor_mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(min);
NTT_DEFINE_TENSOR_BINARY_IMPL(max);
NTT_DEFINE_TENSOR_BINARY_IMPL(pow);
} // namespace nncase::ntt::ops

namespace nncase::ntt::tensor_ops {
namespace detail {
template <class TTensor, template <class T> class Op>
struct tensor_reduce_impl {
    using element_type = typename TTensor::element_type;

    element_type operator()(const TTensor &v) const noexcept {
        Op<element_type> op;
        auto elements = v.elements();
        auto it = elements.begin();
        auto value = *it++;
        for (; it != elements.end(); ++it) {
            value = op(value, *it);
        }
        return value;
    }
};
} // namespace detail

#define NTT_DEFINE_TENSOR_REDUCE_IMPL(op)                                      \
    template <class T, class Shape, class Strides, size_t MaxSize,             \
              bool IsView>                                                     \
    struct reduce<tensor_base<T, Shape, Strides, MaxSize, IsView>, op>         \
        : detail::tensor_reduce_impl<                                          \
              tensor_base<T, Shape, Strides, MaxSize, IsView>, op> {}

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

    TTensor operator()(T value) const noexcept {
        TTensor vec;
        std::fill_n(vec.elements().data(), vec.size(), value);
        return vec;
    }
};

template <class TTensor, template <class T> class Op> struct reduce {
    // scalar
    TTensor operator()(const TTensor &v) const noexcept { return v; }
};

NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::add);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::sub);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::mul);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::div);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::floor_mod);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::mod);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::min);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::max);
NTT_DEFINE_TENSOR_REDUCE_IMPL(ops::pow);
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

template <template <class T> class Op, class TTensor>
auto reduce(const TTensor &tensor) {
    return tensor_ops::reduce<TTensor, Op>()(tensor);
}

template <class TTensor> auto reduce_sum(const TTensor &tensor) {
    return reduce<ops::add>(tensor);
}

template <class TTensor> auto reduce_max(const TTensor &tensor) {
    return reduce<ops::max>(tensor);
}

template <class TTensor> auto reduce_mean(const TTensor &tensor) {
    return div(reduce_sum(tensor), tensor.size());
}
} // namespace nncase::ntt
