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

template <bool Arch> struct u_clamp_policy {
    static constexpr size_t unroll = 2;
};

template <class T1, class T2, bool Arch> struct u_clamp {
  public:
    constexpr void operator()(const T1 *input, size_t input_stride, T1 *output,
                              size_t output_stride, const T2 &min,
                              const T2 &max, size_t count) noexcept {
        using policy_t = u_clamp_policy<Arch>;
        constexpr auto unroll = policy_t::unroll;

        while (count / unroll) {
            for (size_t i = 0; i < unroll; i++) {
                *output = ntt::ops::clamp<T1, T2>()(*input, min, max);
                input += input_stride;
                output += output_stride;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            *output = ntt::ops::clamp<T1, T2>()(*input, min, max);
            input += input_stride;
            output += output_stride;
        }
    }
};
} // namespace ukernels

template <class T1, class T2>
constexpr void u_clamp(const T1 *input, size_t input_stride, T1 *output,
                       size_t output_stride, const T2 &min, const T2 &max,
                       size_t count) noexcept {
    ukernels::u_clamp<T1, T2, true> impl;
    impl(input, input_stride, output, output_stride, min, max, count);
}
} // namespace nncase::ntt
