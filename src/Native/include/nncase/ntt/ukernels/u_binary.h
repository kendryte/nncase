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

template <bool Arch> struct u_binary_policy {
    static constexpr size_t unroll = 2;
};

#define U_BINARY_IMPL(OP)                                                      \
    template <class T1, class T2, class TOut, bool Arch> struct u_##OP {       \
      public:                                                                  \
        constexpr void operator()(const T1 *input1, const T2 *input2,          \
                                  size_t input1_stride, size_t input2_stride,  \
                                  TOut *output, size_t output_stride,          \
                                  size_t count) noexcept {                     \
            using policy_t = u_binary_policy<Arch>;                            \
            constexpr auto unroll = policy_t::unroll;                          \
                                                                               \
            if (count / unroll) {                                              \
                while (count / unroll) {                                       \
                    for (size_t i = 0; i < unroll; i++) {                      \
                        *output = ntt::ops::OP<T1, T2>()(*input1, *input2);    \
                        input1 += input1_stride;                               \
                        input2 += input2_stride;                               \
                        output += output_stride;                               \
                        count--;                                               \
                    }                                                          \
                }                                                              \
            }                                                                  \
                                                                               \
            for (size_t i = 0; i < count; i++) {                               \
                *output = ntt::ops::OP<T1, T2>()(*input1, *input2);            \
                input1 += input1_stride;                                       \
                input2 += input2_stride;                                       \
                output += output_stride;                                       \
            }                                                                  \
        }                                                                      \
    };

U_BINARY_IMPL(add)
U_BINARY_IMPL(div)
U_BINARY_IMPL(max)
U_BINARY_IMPL(min)
U_BINARY_IMPL(mod)
U_BINARY_IMPL(mul)
U_BINARY_IMPL(sub)
} // namespace ukernels

#define U_BINARY(OP)                                                           \
    template <class T1, class T2, class TOut>                                  \
    constexpr void u_##OP(const T1 *input1, const T2 *input2,                  \
                          size_t input1_stride, size_t input2_stride,          \
                          TOut *output, size_t output_stride,                  \
                          size_t count) noexcept {                             \
        ukernels::u_##OP<T1, T2, TOut, true> impl;                             \
        impl(input1, input2, input1_stride, input2_stride, output,             \
             output_stride, count);                                            \
    }
U_BINARY(add)
U_BINARY(div)
U_BINARY(max)
U_BINARY(min)
U_BINARY(mod)
U_BINARY(mul)
U_BINARY(sub)

} // namespace nncase::ntt
