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


template <size_t axis_stride, size_t lane, class T1, class T2, bool Arch>
class u_unpack_1d {
  public:
    constexpr void operator()(const T1 *input, size_t input_stride, T2 *output, size_t count) noexcept {
        using policy_t = u_unpack_policy<T1, T2, Arch>;
        constexpr auto unroll = policy_t::unroll;

        size_t in_offset = 0;
        size_t axis_idx = 0;
        auto extra = (lane - 1) * axis_stride;
        while (count / unroll) {
            for (size_t i = 0; i < unroll; i++) {
                axis_idx = in_offset && (in_offset % axis_stride == 0) ? axis_idx + 1 : axis_idx;
                auto out_offset = in_offset + axis_idx * extra;
                for (size_t j = 0; j < lane; j++)
                    *(output + out_offset + j * axis_stride) = (*input)(j);
                input += input_stride;
                in_offset++;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            axis_idx = in_offset && (in_offset % axis_stride == 0) ? axis_idx + 1 : axis_idx;
            auto out_offset = in_offset + axis_idx * extra;
            for (size_t j = 0; j < lane; j++)
                *(output + out_offset + j * axis_stride) = (*input)(j);
            input += input_stride;
            in_offset++;
        }
    }
};
} // namespace ukernels

template <size_t axis_stride, size_t lane, class T1, class T2>
constexpr void u_unpack_1d(const T1 *input, size_t in_stride, T2 *output, size_t count) noexcept {
    ukernels::u_unpack_1d<axis_stride, lane, T1, T2, true> impl;
    impl(input, in_stride, output, count);
}

} // namespace nncase::ntt
