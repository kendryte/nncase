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
template <template <class T> class Op, class TTensor> struct tensor_unary_impl;

template <template <class T> class Op, IsTensor TTensor>
struct tensor_unary_impl<Op, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr TTensor operator()(const TTensor &v) const noexcept {
        TTensor value;
        apply(v.shape(), [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

  private:
    Op<element_type> op_;
};

template <template <class T> class Op, IsTensor TTensor>
    requires(TTensor::rank() == 2)
struct tensor_unary_impl<Op, TTensor> {
    using sub_vector_type =
        fixed_tensor_alike_t<TTensor, TTensor::shape().at(1)>;

    constexpr TTensor operator()(const TTensor &v) const noexcept {
        TTensor value;
        for (size_t m = 0; m < TTensor::shape().at(0); m++) {
            value(m) = op_(v(m));
        }
        return value;
    }

  private:
    Op<sub_vector_type> op_;
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
            if constexpr (TTensor::shape().rank() == 2 &&
                          T2::shape().rank() == 1) {
                apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(*index.rbegin()));
                });
            } else {
                apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(index));
                });
            }
        } else {
            apply(v1.shape(),
                  [&](auto index) { value(index) = op_(v1(index), v2); });
        }

        return value;
    }

  private:
    Op<element_type1, element_type2> op_;
};

template <template <class T1, class T2> class Op, IsTensor T1, IsTensor T2>
    requires(T1::rank() == 2 && T2::rank() == 2)
struct tensor_binary_impl<Op, T1, T2> {
    using sub_vector_type = fixed_tensor_alike_t<T1, T1::shape().at(1)>;

    constexpr T1 operator()(const T1 &v1, const T2 &v2) const noexcept {
        T1 value;
        for (size_t m = 0; m < T1::shape().at(0); m++) {
            value(m) = op_(v1(m), v2(m));
        }
        return value;
    }

  private:
    Op<sub_vector_type, sub_vector_type> op_;
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

// compare tensor impl
template <template <class T1, class T2> class Op, class T1, class T2>
struct tensor_compare_impl;

template <template <class T1, class T2> class Op, IsTensor TTensor, class T2>
struct tensor_compare_impl<Op, TTensor, T2> {
    using element_type1 = typename TTensor::element_type;
    using element_type2 = element_or_scalar_t<T2>;
    static constexpr size_t vl = TTensor::template lane<0>();
    using TOut = ntt::vector<bool, vl>;
    constexpr TOut operator()(const TTensor &v1, const T2 &v2) const noexcept {
        TOut value;
        if constexpr (IsTensor<T2>) {
            if constexpr (TTensor::shape().rank() == 2 &&
                          T2::shape().rank() == 1) {
                apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(*index.rbegin()));
                });
            } else {
                apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(index));
                });
            }
        } else {
            apply(v1.shape(),
                  [&](auto index) { value(index) = op_(v1(index), v2); });
        }

        return value;
    }

  private:
    Op<element_type1, element_type2> op_;
};

template <template <class T1, class T2> class Op, IsTensor T1, IsTensor T2>
    requires(T1::rank() == 2 && T2::rank() == 2)
struct tensor_compare_impl<Op, T1, T2> {
    using sub_vector_type = fixed_tensor_alike_t<T1, T1::shape().at(1)>;
    static constexpr size_t vl = T1::template lane<0>();
    using TOut = ntt::vector<bool, vl>;
    constexpr TOut operator()(const T1 &v1, const T2 &v2) const noexcept {
        TOut value;
        for (size_t m = 0; m < T1::shape().at(0); m++) {
            value(m) = op_(v1(m), v2(m));
        }
        return value;
    }

  private:
    Op<sub_vector_type, sub_vector_type> op_;
};

template <template <class T1, class T2> class Op, IsScalar TScalar,
          IsTensor TTensor>
struct tensor_compare_impl<Op, TScalar, TTensor> {
    using element_type2 = typename TTensor::element_type;
    static constexpr size_t vl = TTensor::template lane<0>();
    using TOut = ntt::vector<bool, vl>;
    constexpr TOut operator()(const TScalar &v1,
                              const TTensor &v2) const noexcept {
        TOut value;
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

#define NTT_DEFINE_TENSOR_COMPARE_IMPL(op)                                      \
    template <IsTensor T1, class T2>                                           \
    struct op<T1, T2> : detail::tensor_compare_impl<op, T1, T2> {};             \
    template <IsScalar T1, IsTensor T2>                                        \
    struct op<T1, T2> : detail::tensor_compare_impl<op, T1, T2> {}

NTT_DEFINE_TENSOR_UNARY_IMPL(abs);
NTT_DEFINE_TENSOR_UNARY_IMPL(acos);
NTT_DEFINE_TENSOR_UNARY_IMPL(acosh);
NTT_DEFINE_TENSOR_UNARY_IMPL(asin);
NTT_DEFINE_TENSOR_UNARY_IMPL(asinh);
NTT_DEFINE_TENSOR_UNARY_IMPL(ceil);
NTT_DEFINE_TENSOR_UNARY_IMPL(cos);
NTT_DEFINE_TENSOR_UNARY_IMPL(cosh);
NTT_DEFINE_TENSOR_UNARY_IMPL(erf);
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

NTT_DEFINE_TENSOR_COMPARE_IMPL(equal);
NTT_DEFINE_TENSOR_COMPARE_IMPL(not_equal);
NTT_DEFINE_TENSOR_COMPARE_IMPL(greater);
NTT_DEFINE_TENSOR_COMPARE_IMPL(greater_or_equal);
NTT_DEFINE_TENSOR_COMPARE_IMPL(less);
NTT_DEFINE_TENSOR_COMPARE_IMPL(less_or_equal);

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

template <class T1, IsTensor T2, IsTensor T3> struct where<T1, T2, T3> {
    using element_type = typename T2::element_type;

    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T2 value;
        if constexpr (IsTensor<T1>) {
            apply(v1.shape(), [&](auto index) {
                value(index) = op_(condition(index), v1(index), v2(index));
            });
        } else {
            apply(v1.shape(), [&](auto index) {
                value(index) = op_(condition, v1(index), v2(index));
            });
        }

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <class T1, IsScalar T2, IsTensor TTensor>
struct where<T1, T2, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const TTensor &v2) const noexcept {
        TTensor value;
        apply(v2.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1, v2(index));
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <class T1, IsTensor TTensor, IsScalar T2>
struct where<T1, TTensor, T2> {
    using element_type = typename TTensor::element_type;

    constexpr auto operator()(const T1 &condition, const TTensor &v1,
                              const T2 &v2) const noexcept {
        TTensor value;
        apply(v1.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1(index), v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <IsTensor T1, IsScalar T2, IsScalar T3>
struct where<T1, T2, T3> {
    static constexpr size_t vl = T1::template lane<0>();
    using TOut = ntt::vector<T2, vl>;
    using element_type = TOut::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        TOut value;
        apply(condition.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1, v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <IsScalar T1, IsScalar T2, IsTensor T3>
struct where<T1, T2, T3> {
    using element_type = typename T3::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T3 value;
        apply(v2.shape(), [&](auto index) {
            value(index) = op_(condition, v1, v2(index));
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <IsScalar T1, IsTensor T2, IsScalar T3>
struct where<T1, T2, T3> {
    using element_type = typename T2::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T2 value;
        apply(v1.shape(), [&](auto index) {
            value(index) = op_(condition, v1(index), v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <template <class T1, class T2> class Op, IsScalar TResult,
          IsTensor TTensor>
struct reduce<Op, TResult, TTensor> {
    using element_type = typename TTensor::element_type;

    constexpr TResult operator()(const TTensor &v,
                                 TResult init_value) const noexcept {
        Op<TResult, element_type> op;
        auto count = v.shape()[0];
        auto value = init_value;
        for (size_t i = 0; i < count; i++) {
            value = op(value, v(i));
        }
        return value;
    }

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

template <IsTensor TTensor, IsScalar TScalar> struct clamp<TTensor, TScalar> {
    using element_type = typename TTensor::element_type;
    constexpr auto operator()(const TTensor &v, const TScalar &min,
                              const TScalar &max) const noexcept {
        TTensor value;
        apply(v.shape(),
              [&](auto index) { value(index) = op_(v(index), min, max); });
        return value;
    }

  private:
    ops::clamp<element_type, TScalar> op_;
};

template <IsTensor TTensor1, IsTensor TTensor2>
struct cast<TTensor1, TTensor2> {
    using from_type = typename TTensor1::element_type;
    using to_type = typename TTensor2::element_type;
    constexpr auto operator()(const TTensor1 &v) const noexcept {
        TTensor2 value;
        apply(v.shape(), [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

    template <typename... TTensors>
    constexpr auto operator()(const TTensors &...tensors) const noexcept
        requires(sizeof...(tensors) > 1)
    {
        static_assert((... && (std::decay_t<TTensors>::rank() == 1)));

        TTensor2 value;
        size_t count = 0;

        auto process_tensor = [&](const auto &tensor) {
            apply(tensor.shape(),
                  [&](auto index) { value(count++) = op_(tensor(index)); });
        };

        (..., process_tensor(tensors));

        return value;
    }

    constexpr auto operator()(const TTensor1 &v) const noexcept
        requires(IsVector<TTensor1> && (TTensor1::size() != TTensor2::size()))
    {

        static_assert(TTensor1::rank() == 1 && TTensor2::rank() == 1);
        static_assert(ntt::IsVector<TTensor2>);
        static_assert(TTensor2::rank() == 1);
        using value_type2 = typename TTensor2::element_type;
        constexpr auto lanes1 = TTensor1::shape();
        constexpr auto lanes2 = TTensor2::shape();
        constexpr auto type_scale = lanes1[0] / lanes2[0];

        using TOut = ntt::vector<value_type2, type_scale, lanes2[0]>;
        TOut Output;

        size_t count = 0;
        for (size_t i = 0; i < type_scale; i++) {
            apply(Output(i).shape(),
                  [&](auto index) { Output(i)(index) = op_(v(count++)); });
        }

        return Output;
    }

  private:
    ops::cast<from_type, to_type> op_;
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
