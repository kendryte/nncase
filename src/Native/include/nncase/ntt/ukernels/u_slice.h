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

template <bool Arch> struct u_slice_policy {
    static constexpr size_t unroll = 4;
};

template <class T, bool Arch> struct u_slice {
  public:
    constexpr void operator()(const T *pin, size_t in_stride, T *pout,
                              size_t out_stride, size_t count) noexcept {
        using policy_t = u_slice_policy<Arch>;
        constexpr auto unroll = policy_t::unroll;
        while (count / unroll) {
            for (size_t j = 0; j < unroll; j++) {
                *pout = *pin;
                pin += in_stride;
                pout += out_stride;
                count--;
            }
        }

        for (size_t j = 0; j < count; j++) {
            *pout = *pin;
            pin += in_stride;
            pout += out_stride;
        }
    }
};
} // namespace ukernels

template <class T>
void u_slice(const T *pin, size_t in_stride, T *pout, size_t out_stride,
             size_t count) noexcept {
    ukernels::u_slice<T, true> impl;
    impl(pin, in_stride, pout, out_stride, count);
}

} // namespace nncase::ntt
