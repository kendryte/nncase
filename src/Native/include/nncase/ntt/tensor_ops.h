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
#include "utility.h"

namespace nncase::ntt {
// Forward declare get/set elem
template <class TContainer, size_t Rank>
constexpr auto &get_elem(TContainer &&container,
                         ranked_shape<Rank> index) noexcept;

template <class TContainer, class T, size_t Rank>
constexpr void set_elem(const TContainer &container, ranked_shape<Rank> index,
                        T &&value) noexcept;
} // namespace nncase::ntt

namespace nncase::ntt::ops {
// unary_ops ops
namespace detail {
template <template <class T> class Op, IsTensor TTensor>
struct tensor_unary_impl {
    using element_type = typename TTensor::element_type;

    constexpr TTensor operator()(const TTensor &v) const noexcept {
        TTensor value;
        apply(v.shape(), [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

  private:
    Op<element_type> op_;
};

template <template <class T1, class T2> class Op, class T1, class T2>
struct tensor_binary_impl;

template <template <class T1, class T2> class Op, IsTensor TTensor, class T2>
struct tensor_binary_impl<Op, TTensor, T2> {
    using element_type1 = typename TTensor::element_type;
    using element_type2 = element_or_scalar_t<T2>;

    constexpr TTensor operator()(const TTensor &v1,
                                 const T2 &v2) const noexcept {
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

template <template <class T1, class T2> class Op, IsScalar TScalar,
          IsTensor TTensor>
struct tensor_binary_impl<Op, TScalar, TTensor> {
    using element_type2 = typename TTensor::element_type;

    constexpr TTensor operator()(const TScalar &v1,
                                 const TTensor &v2) const noexcept {
        TTensor value;
        apply(v2.shape(),
              [&](auto index) { value(index) = op_(v1, v2(index)); });
        return value;
    }

  private:
    Op<TScalar, element_type2> op_;
};

} // namespace detail

#define NTT_DEFINE_TENSOR_UNARY_IMPL(op)                                       \
    template <IsTensor TTensor>                                                \
    struct op<TTensor> : detail::tensor_unary_impl<op, TTensor> {}

#define NTT_DEFINE_TENSOR_BINARY_IMPL(op)                                      \
    template <IsTensor T1, class T2>                                           \
    struct op<T1, T2> : detail::tensor_binary_impl<op, T1, T2> {};             \
    template <IsScalar T1, IsTensor T2>                                        \
    struct op<T1, T2> : detail::tensor_binary_impl<op, T1, T2> {}

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
NTT_DEFINE_TENSOR_UNARY_IMPL(tanh);
NTT_DEFINE_TENSOR_UNARY_IMPL(swish);

NTT_DEFINE_TENSOR_BINARY_IMPL(add);
NTT_DEFINE_TENSOR_BINARY_IMPL(sub);
NTT_DEFINE_TENSOR_BINARY_IMPL(mul);
NTT_DEFINE_TENSOR_BINARY_IMPL(ceil_div);
NTT_DEFINE_TENSOR_BINARY_IMPL(div);
NTT_DEFINE_TENSOR_BINARY_IMPL(floor_mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(mod);
NTT_DEFINE_TENSOR_BINARY_IMPL(min);
NTT_DEFINE_TENSOR_BINARY_IMPL(max);
NTT_DEFINE_TENSOR_BINARY_IMPL(pow);
NTT_DEFINE_TENSOR_BINARY_IMPL(swishb);

template <IsTensor TTensor> struct inner_product<TTensor, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr auto operator()(const TTensor &v1,
                              const TTensor &v2) const noexcept {
        using result_type = decltype(op_(std::declval<element_type>(),
                                         std::declval<element_type>()));
        result_type value{};
        apply(v1.shape(),
              [&](auto index) { value += op_(v1(index), v2(index)); });
        return value;
    }

  private:
    ops::inner_product<element_type, element_type> op_;
};

template <IsFixedTensor TTensor1, IsFixedTensor TTensor2>
struct outer_product<TTensor1, TTensor2> {
    using element_type = typename TTensor1::element_type;
    static_assert(std::is_same_v<element_type, typename TTensor2::element_type>,
                  "element type not match");

    constexpr auto operator()(const TTensor1 &v1,
                              const TTensor2 &v2) const noexcept {

        using result_type =
            fixed_tensor_alike_t<TTensor1, TTensor1::shape().length(),
                                 TTensor2::shape().length()>;
        result_type value{};
        apply(value.shape(), [&](auto index) {
            value(index) = op_(v1(index[0]), v2(index[1]));
        });
        return value;
    }

  private:
    ops::outer_product<element_type, element_type> op_;
};

template <IsTensor TTensor, class T2> struct mul_add<TTensor, T2, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr auto operator()(const TTensor &v1, const T2 &v2,
                              const TTensor &v3) const noexcept {
        TTensor value;
        if constexpr (IsTensor<T2>) {
            apply(v1.shape(), [&](auto index) {
                value(index) = op_(v1(index), v2(index), v3(index));
            });
        } else {
            apply(v1.shape(), [&](auto index) {
                value(index) = op_(v1(index), v2, v3(index));
            });
        }
        return value;
    }

  private:
    ops::mul_add<element_type, element_type, element_type> op_;
};

template <IsScalar TScalar, IsTensor TTensor>
struct mul_add<TScalar, TTensor, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr auto operator()(const TScalar &s1, const TTensor &v2,
                              const TTensor &v3) const noexcept {
        TTensor value;
        apply(v3.shape(), [&](auto index) {
            value(index) = op_(s1, v2(index), v3(index));
        });
        return value;
    }

  private:
    ops::mul_add<element_type, element_type, element_type> op_;
};

template <template <class T1, class T2> class Op, class TResult,
          IsTensor TTensor>
struct reduce<Op, TResult, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr TResult operator()(const TTensor &v) const noexcept {
        Op<TResult, element_type> op;
        auto count = v.shape()[0];
        auto value = v(0);
        for (size_t i = 1; i < count; i++) {
            value = op(value, v(i));
        }
        return value;
    }
};
} // namespace nncase::ntt::ops

namespace nncase::ntt::tensor_ops {
template <class TTensor> struct tload {
    using T = typename TTensor::element_type;

    constexpr TTensor operator()(const T *src) const noexcept {
        TTensor vec;
        std::copy(src, src + vec.size(), vec.buffer().data());
        return vec;
    }
};

template <class TTensor> struct tload_scalar {
    using T = typename TTensor::element_type;

    constexpr TTensor operator()(const T &value) const noexcept {
        TTensor vec;
        std::fill_n(vec.buffer().data(), vec.size(), value);
        return vec;
    }
};
} // namespace nncase::ntt::tensor_ops

namespace nncase::ntt {
template <class T, class Shape, class Strides, size_t MaxSize>
detail::tensor_impl<T, Shape, Strides, MaxSize, false, true>::tensor_impl(
    T value) noexcept
    : tensor_impl(
          basic_tensor<T, Shape, Strides, MaxSize, false>::from_scalar(value)) {
}

template <class T, class Shape, class Strides, size_t MaxSize, bool IsView>
basic_tensor<T, Shape, Strides, MaxSize, IsView>
basic_tensor<T, Shape, Strides, MaxSize, IsView>::from_scalar(
    T value) noexcept {
    return tensor_ops::tload_scalar<
        basic_tensor<T, Shape, Strides, MaxSize, false>>()(value);
}

template <class T, size_t... Lanes>
basic_vector<T, Lanes...> basic_vector<T, Lanes...>::from_scalar(T v) noexcept {
    return tensor_ops::tload_scalar<basic_vector<T, Lanes...>>()(v);
}

template <IsTensor TTensor, IsScalar T>
constexpr TTensor tload(const T *src) noexcept {
    return tensor_ops::tload<TTensor>()(src);
}
} // namespace nncase::ntt
