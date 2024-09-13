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
#include "tensor.h"
#include "tensor_traits.h"

namespace nncase::ntt::ukernels {
template <size_t M, size_t N, size_t MStrides, bool Arch, class TIn, class TOut>
class u_pack {
  public:
    constexpr void operator()(const TIn *input, TOut *output) noexcept {
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                output[j](i) = input[i * MStrides + j];
            }
        }

        if constexpr (M < TOut::shape_type::length()) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = M; i < TOut::shape_type::length(); i++) {
                    output[j](i) = (TIn)0;
                }
            }
        }
    }
};

template <reduce_op Op> struct reduce_to_binary_type;

template <> struct reduce_to_binary_type<reduce_op::mean> {
    template <class T1, class T2> using type = ops::add<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::min> {
    template <class T1, class T2> using type = ops::min<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::max> {
    template <class T1, class T2> using type = ops::max<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::sum> {
    template <class T1, class T2> using type = ops::add<T1, T2>;
};

template <> struct reduce_to_binary_type<reduce_op::prod> {
    template <class T1, class T2> using type = ops::mul<T1, T2>;
};

template <reduce_op Op, class T, bool Arch> struct u_reduce_policy {
    static constexpr size_t unroll = 1;
};

template <reduce_op Op, class T, bool Arch> struct u_reduce {
  public:
    constexpr T operator()(const T *input, size_t input_stride, size_t count,
                           T init_value) noexcept {
        using binary_op_t =
            typename reduce_to_binary_type<Op>::template type<T, T>;
        using policy_t = u_reduce_policy<Op, T, Arch>;
        constexpr auto unroll = policy_t::unroll;

        if (count / unroll) {
            T temp[unroll];
#if 1
            for (size_t i = 0; i < unroll; i++) {
                temp[i] = *input;
                input += input_stride;
                count--;
            }

            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    temp[i] = binary_op_t()(temp[i], *input);
                    input += input_stride;
                    count--;
                }
            }

            init_value = binary_op_t()(init_value, tree_reduce<unroll>(temp));
#else
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    temp[i] = *input;
                    input += input_stride;
                    count--;
                }
                init_value =
                    binary_op_t()(init_value, tree_reduce<unroll>(temp));
            }
#endif
        }

        for (size_t i = 0; i < count; i++) {
            init_value = binary_op_t()(init_value, *input);
            input += input_stride;
        }
        return init_value;
    }

    template <size_t N> constexpr T tree_reduce(T *input) noexcept {
        using binary_op_t =
            typename reduce_to_binary_type<Op>::template type<T, T>;
        if constexpr (N == 2) {
            return binary_op_t()(input[0], input[1]);
        } else {
            return binary_op_t()(tree_reduce<N / 2>(input),
                                 tree_reduce<N / 2>(input + N / 2));
        }
    }
};
} // namespace nncase::ntt::ukernels

namespace nncase::ntt {
template <size_t M, size_t N, size_t MStrides, class TIn, class TOut>
constexpr void u_pack(const TIn *input, TOut *output) noexcept {
    ukernels::u_pack<M, N, MStrides, true, std::decay_t<TIn>,
                     std::decay_t<TOut>>
        impl;
    impl(input, output);
}

template <reduce_op Op, class T>
constexpr T u_reduce(const T *input, size_t input_stride, size_t count,
                     T init_value) {
    ukernels::u_reduce<Op, T, true> impl;
    return impl(input, input_stride, count, init_value);
}
} // namespace nncase::ntt
