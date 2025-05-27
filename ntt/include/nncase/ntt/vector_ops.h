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
#include "apply.h"
#include "primitive_ops.h"
#include "tensor_traits.h"
#include "vector.h"

namespace nncase::ntt::ops {
// unary_ops ops
namespace detail {
template <template <class T> class Op, class TVector> struct tensor_unary_impl;

template <template <class T> class Op, Vector TVector>
struct tensor_unary_impl<Op, TVector> {
    using element_type = typename TVector::element_type;

    constexpr TVector operator()(const TVector &v) const noexcept {
        TVector value;
        ntt::apply(v.shape(),
                   [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

  private:
    Op<element_type> op_;
};

template <template <class T> class Op, Vector TVector>
    requires(TVector::rank() == 2)
struct tensor_unary_impl<Op, TVector> {
    using sub_vector_type =
        vector<typename TVector::element_type, TVector::shape().at(1)>;

    constexpr TVector operator()(const TVector &v) const noexcept {
        TVector value;
        for (size_t m = 0; m < TVector::shape().at(0); m++) {
            value(m) = op_(v(m));
        }
        return value;
    }

  private:
    Op<sub_vector_type> op_;
};

template <template <class T1, class T2> class Op, class T1, class T2>
struct tensor_binary_impl;

template <template <class T1, class T2> class Op, Vector TVector, class T2>
struct tensor_binary_impl<Op, TVector, T2> {
    using element_type1 = typename TVector::element_type;
    using element_type2 = element_or_scalar_t<T2>;

    constexpr TVector operator()(const TVector &v1,
                                 const T2 &v2) const noexcept {
        TVector value;
        if constexpr (Vector<T2>) {
            if constexpr (TVector::rank() == 2 && T2::rank() == 1) {
                ntt::apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(index[1_dim]));
                });
            } else {
                ntt::apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(index));
                });
            }
        } else {
            ntt::apply(v1.shape(),
                       [&](auto index) { value(index) = op_(v1(index), v2); });
        }

        return value;
    }

  private:
    Op<element_type1, element_type2> op_;
};

template <template <class T1, class T2> class Op, Vector T1, Vector T2>
    requires(T1::rank() == 2 && T2::rank() == 2)
struct tensor_binary_impl<Op, T1, T2> {
    using sub_vector_type =
        vector<typename T1::element_type, T1::shape().at(1)>;

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

template <template <class T1, class T2> class Op, Scalar TScalar,
          Vector TVector>
struct tensor_binary_impl<Op, TScalar, TVector> {
    using element_type2 = typename TVector::element_type;

    constexpr TVector operator()(const TScalar &v1,
                                 const TVector &v2) const noexcept {
        TVector value;
        ntt::apply(v2.shape(),
                   [&](auto index) { value(index) = op_(v1, v2(index)); });
        return value;
    }

  private:
    Op<TScalar, element_type2> op_;
};

// compare tensor impl
template <template <class T1, class T2> class Op, class T1, class T2>
struct tensor_compare_impl;

template <template <class T1, class T2> class Op, Vector TVector, class T2>
struct tensor_compare_impl<Op, TVector, T2> {
    using element_type1 = typename TVector::element_type;
    using element_type2 = element_or_scalar_t<T2>;
    static constexpr size_t vl = TVector::template lane<0>();
    using TOut = ntt::vector<bool, vl>;
    constexpr TOut operator()(const TVector &v1, const T2 &v2) const noexcept {
        TOut value;
        if constexpr (Vector<T2>) {
            if constexpr (TVector::shape().rank() == 2 &&
                          T2::shape().rank() == 1) {
                ntt::apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(*index.rbegin()));
                });
            } else {
                ntt::apply(v1.shape(), [&](auto index) {
                    value(index) = op_(v1(index), v2(index));
                });
            }
        } else {
            ntt::apply(v1.shape(),
                       [&](auto index) { value(index) = op_(v1(index), v2); });
        }

        return value;
    }

  private:
    Op<element_type1, element_type2> op_;
};

template <template <class T1, class T2> class Op, Vector T1, Vector T2>
    requires(T1::rank() == 2 && T2::rank() == 2)
struct tensor_compare_impl<Op, T1, T2> {
    using sub_vector_type =
        vector<typename T1::element_type, T1::shape().at(1)>;
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

template <template <class T1, class T2> class Op, Scalar TScalar,
          Vector TVector>
struct tensor_compare_impl<Op, TScalar, TVector> {
    using element_type2 = typename TVector::element_type;
    static constexpr size_t vl = TVector::template lane<0>();
    using TOut = ntt::vector<bool, vl>;
    constexpr TOut operator()(const TScalar &v1,
                              const TVector &v2) const noexcept {
        TOut value;
        ntt::apply(v2.shape(),
                   [&](auto index) { value(index) = op_(v1, v2(index)); });
        return value;
    }

  private:
    Op<TScalar, element_type2> op_;
};

} // namespace detail

#define NTT_DEFINE_TENSOR_UNARY_IMPL(op)                                       \
    template <Vector TVector>                                                  \
    struct op<TVector> : detail::tensor_unary_impl<op, TVector> {}

#define NTT_DEFINE_TENSOR_BINARY_IMPL(op)                                      \
    template <Vector T1, class T2>                                             \
    struct op<T1, T2> : detail::tensor_binary_impl<op, T1, T2> {};             \
    template <Scalar T1, Vector T2>                                            \
    struct op<T1, T2> : detail::tensor_binary_impl<op, T1, T2> {}

#define NTT_DEFINE_TENSOR_COMPARE_IMPL(op)                                     \
    template <Vector T1, class T2>                                             \
    struct op<T1, T2> : detail::tensor_compare_impl<op, T1, T2> {};            \
    template <Scalar T1, Vector T2>                                            \
    struct op<T1, T2> : detail::tensor_compare_impl<op, T1, T2> {}

NTT_DEFINE_TENSOR_UNARY_IMPL(abs);
NTT_DEFINE_TENSOR_UNARY_IMPL(acos);
NTT_DEFINE_TENSOR_UNARY_IMPL(acosh);
NTT_DEFINE_TENSOR_UNARY_IMPL(asin);
NTT_DEFINE_TENSOR_UNARY_IMPL(asinh);
NTT_DEFINE_TENSOR_UNARY_IMPL(ceil);
NTT_DEFINE_TENSOR_UNARY_IMPL(copy);
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

template <Vector TVector> struct inner_product<TVector, TVector> {
    using element_type = typename TVector::element_type;

    constexpr auto operator()(const TVector &v1,
                              const TVector &v2) const noexcept {
        using result_type = decltype(op_(std::declval<element_type>(),
                                         std::declval<element_type>()));
        result_type value{};
        ntt::apply(v1.shape(),
                   [&](auto index) { value += op_(v1(index), v2(index)); });
        return value;
    }

  private:
    ops::inner_product<element_type, element_type> op_;
};

template <FixedTensor TVector1, FixedTensor TVector2>
struct outer_product<TVector1, TVector2> {
    using element_type = typename TVector1::element_type;
    static_assert(std::is_same_v<element_type, typename TVector2::element_type>,
                  "element type not match");

    constexpr auto operator()(const TVector1 &v1,
                              const TVector2 &v2) const noexcept {

        using result_type =
            vector<typename TVector1::element_type, TVector1::shape().length(),
                   TVector2::shape().length()>;
        result_type value{};
        ntt::apply(value.shape(), [&](auto index) {
            value(index) = op_(v1(index[0]), v2(index[1]));
        });
        return value;
    }

  private:
    ops::outer_product<element_type, element_type> op_;
};

template <Vector TVector, class T2> struct mul_add<TVector, T2, TVector> {
    using element_type = typename TVector::element_type;

    constexpr auto operator()(const TVector &v1, const T2 &v2,
                              const TVector &v3) const noexcept {
        TVector value;
        if constexpr (Vector<T2>) {
            ntt::apply(v1.shape(), [&](auto index) {
                value(index) = op_(v1(index), v2(index), v3(index));
            });
        } else {
            ntt::apply(v1.shape(), [&](auto index) {
                value(index) = op_(v1(index), v2, v3(index));
            });
        }
        return value;
    }

  private:
    ops::mul_add<element_type, element_type, element_type> op_;
};

template <Scalar TScalar, Vector TVector>
struct mul_add<TScalar, TVector, TVector> {
    using element_type = typename TVector::element_type;

    constexpr auto operator()(const TScalar &s1, const TVector &v2,
                              const TVector &v3) const noexcept {
        TVector value;
        ntt::apply(v3.shape(), [&](auto index) {
            value(index) = op_(s1, v2(index), v3(index));
        });
        return value;
    }

  private:
    ops::mul_add<element_type, element_type, element_type> op_;
};

template <class T1, Vector T2, Vector T3> struct where<T1, T2, T3> {
    using element_type = typename T2::element_type;

    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T2 value;
        if constexpr (Vector<T1>) {
            ntt::apply(v1.shape(), [&](auto index) {
                value(index) = op_(condition(index), v1(index), v2(index));
            });
        } else {
            ntt::apply(v1.shape(), [&](auto index) {
                value(index) = op_(condition, v1(index), v2(index));
            });
        }

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <class T1, Scalar T2, Vector TVector> struct where<T1, T2, TVector> {
    using element_type = typename TVector::element_type;

    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const TVector &v2) const noexcept {
        TVector value;
        ntt::apply(v2.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1, v2(index));
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <class T1, Vector TVector, Scalar T2> struct where<T1, TVector, T2> {
    using element_type = typename TVector::element_type;

    constexpr auto operator()(const T1 &condition, const TVector &v1,
                              const T2 &v2) const noexcept {
        TVector value;
        ntt::apply(v1.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1(index), v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <Vector T1, Scalar T2, Scalar T3> struct where<T1, T2, T3> {
    static constexpr size_t vl = T1::template lane<0>();
    using TOut = ntt::vector<T2, vl>;
    using element_type = TOut::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        TOut value;
        ntt::apply(condition.shape(), [&](auto index) {
            value(index) = op_(condition(index), v1, v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <Scalar T1, Scalar T2, Vector T3> struct where<T1, T2, T3> {
    using element_type = typename T3::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T3 value;
        ntt::apply(v2.shape(), [&](auto index) {
            value(index) = op_(condition, v1, v2(index));
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <Scalar T1, Vector T2, Scalar T3> struct where<T1, T2, T3> {
    using element_type = typename T2::element_type;
    constexpr auto operator()(const T1 &condition, const T2 &v1,
                              const T3 &v2) const noexcept {
        T2 value;
        ntt::apply(v1.shape(), [&](auto index) {
            value(index) = op_(condition, v1(index), v2);
        });

        return value;
    }

  private:
    ops::where<bool, element_type, element_type> op_;
};

template <template <class T1, class T2> class Op, Scalar TResult,
          Vector TVector>
struct reduce<Op, TResult, TVector> {
    using element_type = typename TVector::element_type;

    constexpr TResult operator()(const TVector &v,
                                 TResult init_value) const noexcept {
        Op<TResult, element_type> op;
        auto count = v.shape()[0];
        auto value = init_value;
        for (size_t i = 0; i < count; i++) {
            value = op(value, v(i));
        }
        return value;
    }

    constexpr TResult operator()(const TVector &v) const noexcept {
        Op<TResult, element_type> op;
        auto count = v.shape()[0];
        auto value = v(0);
        for (size_t i = 1; i < count; i++) {
            value = op(value, v(i));
        }
        return value;
    }
};

template <Vector TVector, Scalar TScalar> struct clamp<TVector, TScalar> {
    using element_type = typename TVector::element_type;
    constexpr auto operator()(const TVector &v, const TScalar &min,
                              const TScalar &max) const noexcept {
        TVector value;
        ntt::apply(v.shape(),
                   [&](auto index) { value(index) = op_(v(index), min, max); });
        return value;
    }

  private:
    ops::clamp<element_type, TScalar> op_;
};

template <Vector TVector1, Vector TVector2> struct cast<TVector1, TVector2> {
    using from_type = typename TVector1::element_type;
    using to_type = typename TVector2::element_type;
    constexpr auto operator()(const TVector1 &v) const noexcept {
        TVector2 value;
        ntt::apply(v.shape(),
                   [&](auto index) { value(index) = op_(v(index)); });
        return value;
    }

    template <typename... TVectors>
    constexpr auto operator()(const TVectors &...tensors) const noexcept
        requires(sizeof...(tensors) > 1)
    {
        static_assert((... && (std::decay_t<TVectors>::rank() == 1)));

        TVector2 value;
        size_t count = 0;

        auto process_tensor = [&](const auto &tensor) {
            ntt::apply(tensor.shape(), [&](auto index) {
                value(count++) = op_(tensor(index));
            });
        };

        (..., process_tensor(tensors));

        return value;
    }

    constexpr auto operator()(const TVector1 &v) const noexcept
        requires(Vector<TVector1> && (TVector1::size() != TVector2::size()))
    {

        static_assert(TVector1::rank() == 1 && TVector2::rank() == 1);
        static_assert(ntt::Vector<TVector2>);
        static_assert(TVector2::rank() == 1);
        using value_type2 = typename TVector2::element_type;
        constexpr auto lanes1 = TVector1::shape();
        constexpr auto lanes2 = TVector2::shape();
        constexpr auto type_scale = lanes1[0] / lanes2[0];

        using TOut = ntt::vector<value_type2, type_scale, lanes2[0]>;
        TOut Output;

        size_t count = 0;
        for (size_t i = 0; i < type_scale; i++) {
            ntt::apply(Output(i).shape(),
                       [&](auto index) { Output(i)(index) = op_(v(count++)); });
        }

        return Output;
    }

  private:
    ops::cast<from_type, to_type> op_;
};

} // namespace nncase::ntt::ops

namespace nncase::ntt::vector_ops {
template <Vector TVector> struct vload_scalar {
    using T = typename TVector::element_type;

    constexpr TVector operator()(const T &value) const noexcept {
        TVector vec;
        std::fill_n(vec.buffer().data(), vec.size(), value);
        return vec;
    }
};

template <bool AccC, bool TransA, Vector T1, Vector T2, Vector TResult>
struct vmma {
    constexpr TResult operator()(const T1 &lhs, const T2 &rhs,
                                 const TResult &v3) const noexcept {
        static_assert(T1::rank() == T2::rank() &&
                          T2::rank() == TResult::rank() && TResult::rank() == 2,
                      "only support 2d mma");
        TResult output = v3;
        if constexpr (TransA) {
            // <k,m> @ <k,n>
            if constexpr (AccC) {
                output = ntt::outer_product(lhs(0), rhs(0)) + output;
            } else {
                output = ntt::outer_product(lhs(0), rhs(0));
            }

            for (size_t k = 1; k < T1::shape().at(0); k++) {
                output = ntt::outer_product(lhs(k), rhs(k)) + output;
            }
        } else {
            for (size_t k = 0; k < T2::shape().at(0); k++) {
                for (size_t m = 0; m < T1::shape().at(0); m++) {
                    output(m) = (k != 0 || AccC)
                                    ? ntt::mul_add(lhs(m, k), rhs(k), output(m))
                                    : ntt::mul(lhs(m, k), rhs(k));
                }
            }
        }

        return output;
    }
};
} // namespace nncase::ntt::vector_ops

namespace nncase::ntt {
template <Scalar T, size_t... Lanes>
basic_vector<T, Lanes...>
basic_vector<T, Lanes...>::from_scalar(T value) noexcept {
    return vector_ops::vload_scalar<basic_vector<T, Lanes...>>()(value);
}

template <bool AccC, bool TransA, Vector T1, Vector T2, Vector TResult>
constexpr TResult vmma(const T1 &v1, const T2 &v2, const TResult &v3) noexcept {
    return vector_ops::vmma<AccC, TransA, T1, T2, TResult>()(v1, v2, v3);
}
} // namespace nncase::ntt
