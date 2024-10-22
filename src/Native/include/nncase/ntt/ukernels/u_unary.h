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

template <bool Arch> struct u_unary_policy {
    static constexpr size_t unroll = 2;
};

#define U_UNARY_IMPL(OP)                                                       \
    template <class T1, class T2, bool Arch> struct u_##OP {                   \
      public:                                                                  \
        constexpr void operator()(const T1 *input, size_t input_stride,        \
                                  T2 *output, size_t output_stride,            \
                                  size_t count) noexcept {                     \
            using policy_t = u_unary_policy<Arch>;                             \
            constexpr auto unroll = policy_t::unroll;                          \
                                                                               \
            if (count / unroll) {                                              \
                while (count / unroll) {                                       \
                    for (size_t i = 0; i < unroll; i++) {                      \
                        *output = ntt::ops::OP<T1>()(*input);                  \
                        input += input_stride;                                 \
                        output += output_stride;                               \
                        count--;                                               \
                    }                                                          \
                }                                                              \
            }                                                                  \
                                                                               \
            for (size_t i = 0; i < count; i++) {                               \
                *output = ntt::ops::OP<T1>()(*input);                          \
                input += input_stride;                                         \
                output += output_stride;                                       \
            }                                                                  \
        }                                                                      \
    };

U_UNARY_IMPL(ceil)
U_UNARY_IMPL(abs)
U_UNARY_IMPL(floor)
U_UNARY_IMPL(neg)
U_UNARY_IMPL(round)
U_UNARY_IMPL(sign)
U_UNARY_IMPL(square)
} // namespace ukernels

#define U_UNARY(OP)                                                            \
    template <class T1, class T2>                                              \
    constexpr void u_##OP(const T1 *input, size_t input_stride, T2 *output,    \
                          size_t output_stride, size_t count) noexcept {       \
        ukernels::u_##OP<T1, T2, true> impl;                                   \
        impl(input, input_stride, output, output_stride, count);               \
    }
U_UNARY(ceil)
U_UNARY(abs)
U_UNARY(floor)
U_UNARY(neg)
U_UNARY(round)
U_UNARY(sign)
U_UNARY(square)
} // namespace nncase::ntt
