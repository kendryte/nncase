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
#include "../primitive_ops.h"

namespace nncase::ntt {
namespace ukernels {
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
    static constexpr size_t unroll = 2;
};

template <Tensor TIn, Tensor TOut, FixedDimensions TReduceAxes,
          FixedDimensions VectorizedAxes, bool Arch>
struct u_reduce_2d {
  public:
    constexpr void operator()(const TIn &input, TOut &output,
                              const TReduceAxes &,
                              const VectorizedAxes &) noexcept {
        using TInElem = typename TIn::element_type;
        constexpr auto reduce_axes = TReduceAxes{};
        ntt::apply(output.shape(), [&](auto index) {
            TInElem reduced_in = 0;
            for (size_t i = 0; i < input.shape()[reduce_axes.at(0)]; i++) {
                auto in_index = generate_shape<TIn::rank()>([&](auto j) {
                    if (j == reduce_axes.at(0))
                        return (dim_t)i;
                    else
                        return (dim_t)index[j];
                });
                reduced_in += input(in_index);
            }
            output(index) = reduced_in;
        });
    }
};

template <reduce_op Op, class T, bool Arch> struct u_reduce {
  public:
    constexpr T operator()(const T *input, size_t input_stride, size_t count,
                           T init_value) noexcept {
        using binary_op_t =
            typename reduce_to_binary_type<Op>::template type<T, T>;
        using policy_t = u_reduce_policy<Op, T, Arch>;
        constexpr auto unroll = policy_t::unroll;
        __asm__ volatile("" ::: "memory");

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
} // namespace ukernels

template <reduce_op Op, class T>
constexpr T u_reduce(const T *input, size_t input_stride, size_t count,
                     T init_value) noexcept {
    ukernels::u_reduce<Op, T, true> impl;
    return impl(input, input_stride, count, init_value);
}

template <Tensor TIn, Tensor TOut, FixedDimensions TReduceAxes,
          FixedDimensions VectorizedAxes>
constexpr void u_reduce(const TIn &input, TOut &output, const TReduceAxes &,
                        const VectorizedAxes &) noexcept {
    ukernels::u_reduce_2d<TIn, TOut, TReduceAxes, VectorizedAxes, true> impl;
    impl(input, output, TReduceAxes{}, VectorizedAxes{});
}
} // namespace nncase::ntt
