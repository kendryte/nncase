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
#include "../loop.h"
#include <cstddef>
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {

template <class T1, class T2, bool Arch> struct u_unpack_policy {
    static constexpr size_t unroll = 2;
};

template <size_t axis_stride, size_t lane, class T1, class T2, bool Arch,
          size_t PackAxis>
class u_unpack_1d_fixed {
  public:
    void operator()(const T1 &input, size_t input_stride, T2 *output,
                    size_t count) noexcept {
        auto in_ptr = input.buffer().data();
        using policy_t = u_unpack_policy<T1, T2, Arch>;
        constexpr auto unroll = policy_t::unroll;
        size_t in_offset = 0;
        size_t axis_idx = 0;
        auto extra = (lane - 1) * axis_stride;
        while (count / axis_stride) {
            auto tmp = axis_stride;
            while (tmp / unroll) {
                auto out_ptr = output + in_offset;
                for (size_t i = 0; i < unroll; i++) {
                    auto out = out_ptr + i + axis_idx * extra;
                    for (size_t j = 0; j < lane; j++)
                        *(out + j * axis_stride) = (*in_ptr)(j);
                    in_ptr += input_stride;
                }
                in_offset += unroll;
                count -= unroll;
                tmp -= unroll;
            }

            for (size_t i = 0; i < tmp; i++) {
                auto out = output + in_offset + axis_idx * extra;
                for (size_t j = 0; j < lane; j++)
                    *(out + j * axis_stride) = (*in_ptr)(j);
                in_ptr += input_stride;
                in_offset++;
                count--;
            }
            axis_idx++;
        }
    }
};

template <size_t low_axis_stride, size_t low_lane, size_t high_axis_stride,
          size_t high_lane, class T1, class T2, bool Arch>
class u_unpack_2d_fixed {
  public:
    void operator()(const T1 *input, size_t input_stride, T2 *output,
                    size_t count) noexcept {
        using policy_t = u_unpack_policy<T1, T2, Arch>;
        constexpr auto unroll = policy_t::unroll;
        size_t in_offset = 0;
        size_t low_idx = 0;
        size_t high_idx = 0;
        constexpr auto high_dim = low_axis_stride / high_axis_stride;
        constexpr auto out_low_strides = low_axis_stride * low_lane;
        constexpr auto low_extra = low_axis_stride * (low_lane * low_lane - 1);
        constexpr auto high_extra = high_axis_stride * (high_lane - 1);

        while (count / high_axis_stride) {
            auto out_ptr = output + in_offset + low_idx * low_extra +
                           high_idx * high_extra;
            auto out_end = out_ptr + high_axis_stride;
            while (out_ptr < out_end) {
                auto tmp = low_lane;
                size_t i_idx = 0;
                while (tmp / unroll) {
                    for (size_t i = 0; i < unroll; i++) {
                        for (size_t j = 0; j < high_lane; j++)
                            *(out_ptr + i_idx * out_low_strides +
                              j * high_axis_stride) = (*input)(i_idx)(j);
                        i_idx++;
                    }
                    tmp -= unroll;
                }

                for (; i_idx < low_lane; i_idx++) {
                    for (size_t j = 0; j < high_lane; j++)
                        *(out_ptr + i_idx * out_low_strides +
                          j * high_axis_stride) = (*input)(i_idx)(j);
                }

                input += input_stride;
                out_ptr += 1;
            }
            in_offset += high_axis_stride;
            count -= high_axis_stride;
            high_idx++;
            if (high_idx == high_dim) {
                high_idx = 0;
                low_idx++;
            }
        }
    }
};

template <size_t lane, class T1, class T2, bool Arch> class u_unpack_1d_ranked {
  public:
    void operator()(const T1 *input, size_t input_stride, size_t axis_stride,
                    T2 *output, size_t count) noexcept {
        using policy_t = u_unpack_policy<T1, T2, Arch>;
        constexpr auto unroll = policy_t::unroll;
        size_t in_offset = 0;
        size_t axis_idx = 0;
        auto extra = (lane - 1) * axis_stride;
        while (count / axis_stride) {
            auto tmp = axis_stride;
            while (tmp / unroll) {
                auto out_ptr = output + in_offset;
                for (size_t i = 0; i < unroll; i++) {
                    auto out = out_ptr + i + axis_idx * extra;
                    for (size_t j = 0; j < lane; j++)
                        *(out + j * axis_stride) = (*input)(j);
                    input += input_stride;
                }
                in_offset += unroll;
                count -= unroll;
                tmp -= unroll;
            }

            for (size_t i = 0; i < tmp; i++) {
                auto out = output + in_offset + axis_idx * extra;
                for (size_t j = 0; j < lane; j++)
                    *(out + j * axis_stride) = (*input)(j);
                input += input_stride;
                in_offset++;
                count--;
            }
            axis_idx++;
        }
    }
};
} // namespace ukernels

template <size_t axis_stride, size_t lane, class T1, class T2, size_t PackAxis>
void u_unpack_1d_fixed(const T1 &input, size_t in_stride, T2 *output,
                       size_t count) noexcept {
    ukernels::u_unpack_1d_fixed<axis_stride, lane, T1, T2, true, PackAxis> impl;
    impl(input, in_stride, output, count);
}

template <size_t low_axis_stride, size_t low_lane, size_t high_axis_stride,
          size_t high_lane, class T1, class T2>
void u_unpack_2d_fixed(const T1 *input, size_t in_stride, T2 *output,
                       size_t count) noexcept {
    ukernels::u_unpack_2d_fixed<low_axis_stride, low_lane, high_axis_stride,
                                high_lane, T1, T2, true>
        impl;
    impl(input, in_stride, output, count);
}

template <size_t lane, class T1, class T2>
void u_unpack_1d_ranked(const T1 *input, size_t in_stride, size_t axis_stride,
                        T2 *output, size_t count) noexcept {
    ukernels::u_unpack_1d_ranked<lane, T1, T2, true> impl;
    impl(input, in_stride, axis_stride, output, count);
}
} // namespace nncase::ntt
