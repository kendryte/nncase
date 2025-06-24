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
#include "../vector.h"

namespace nncase::ntt {
namespace ukernels {

template <bool Arch> struct u_cast_policy {
    static constexpr size_t unroll = 2;
};

template <class T1, class T2, bool Arch, size_t in_offset_scale,
          size_t out_offset_scale>
struct u_cast {
  public:
    constexpr void operator()(const T1 *input, size_t input_stride, T2 *output,
                              size_t output_stride, size_t count) noexcept {
        using policy_t = u_cast_policy<Arch>;
        constexpr auto unroll = policy_t::unroll;

        if constexpr (in_offset_scale == 4 && out_offset_scale == 1) {
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    *output =
                        ntt::ops::cast<T1, T2>()(*(input + 0 * input_stride),
                                                 *(input + 1 * input_stride),
                                                 *(input + 2 * input_stride),
                                                 *(input + 3 * input_stride));
                    input += input_stride * in_offset_scale;
                    output += output_stride * out_offset_scale;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                *output = ntt::ops::cast<T1, T2>()(
                    *(input + 0 * input_stride), *(input + 1 * input_stride),
                    *(input + 2 * input_stride), *(input + 3 * input_stride));
                input += input_stride * in_offset_scale;
                output += output_stride * out_offset_scale;
            }
        } else if constexpr (in_offset_scale == 2 && out_offset_scale == 1) {
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    *output =
                        ntt::ops::cast<T1, T2>()(*(input + 0 * input_stride),
                                                 *(input + 1 * input_stride));
                    input += input_stride * in_offset_scale;
                    output += output_stride * out_offset_scale;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                *output = ntt::ops::cast<T1, T2>()(*(input + 0 * input_stride),
                                                   *(input + 1 * input_stride));
                input += input_stride * in_offset_scale;
                output += output_stride * out_offset_scale;
            }
        } else if constexpr (in_offset_scale == 1 && out_offset_scale > 1) {
            using value_type = typename T2::element_type;
            constexpr auto lanes = T2::shape();
            using TOut = ntt::vector<value_type, out_offset_scale, lanes[0]>;

            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    *((TOut *)output) = ntt::ops::cast<T1, T2>()(*input);
                    input += input_stride * in_offset_scale;
                    output += output_stride * out_offset_scale;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                *((TOut *)output) = ntt::ops::cast<T1, T2>()(*input);
                input += input_stride * in_offset_scale;
                output += output_stride * out_offset_scale;
            }

        } else {
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    *output = ntt::ops::cast<T1, T2>()(*input);
                    input += input_stride * in_offset_scale;
                    output += output_stride * out_offset_scale;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                *output = ntt::ops::cast<T1, T2>()(*input);
                input += input_stride * in_offset_scale;
                output += output_stride * out_offset_scale;
            }
        }
    }
};
} // namespace ukernels

template <class T1, class T2, size_t in_offset_scale, size_t out_offset_scale>
constexpr void u_cast(const T1 *input, size_t input_stride, T2 *output,
                      size_t output_stride, size_t count) noexcept {
    ukernels::u_cast<T1, T2, true, in_offset_scale, out_offset_scale> impl;
    impl(input, input_stride, output, output_stride, count);
}
} // namespace nncase::ntt
