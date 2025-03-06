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

template <class Op, class T1, class T2, class T3, bool Arch>
struct u_where_policy {
    static constexpr size_t unroll = 1;
};

template <class Op, class T1, class T2, class T3, class TOut, bool Arch>
struct u_where {
  public:
    constexpr void operator()(const T1 *cond, size_t cond_stride, const T2 *x,
                              size_t x_stride, const T3 *y, size_t y_stride,
                              TOut *output, size_t output_stride,
                              size_t count) noexcept {
        using policy_t = u_where_policy<Op, T1, T2, T3, Arch>;
        constexpr auto unroll = policy_t::unroll;
        Op op;

        while (count >= unroll) {
            for (size_t i = 0; i < unroll; i++) {
                *output = op(*cond, *x, *y);
                cond += cond_stride;
                x += x_stride;
                y += y_stride;
                output += output_stride;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            *output = op(*cond, *x, *y);
            cond += cond_stride;
            x += x_stride;
            y += y_stride;
            output += output_stride;
        }
    }
};
} // namespace ukernels

template <class Op, class T1, class T2, class T3, class TOut>
constexpr void u_where(const T1 *cond, size_t cond_stride, const T2 *x,
                       size_t x_stride, const T3 *y, size_t y_stride,
                       TOut *output, size_t output_stride,
                       size_t count) noexcept {
    ukernels::u_where<Op, T1, T2, T3, TOut, true> impl;
    impl(cond, cond_stride, x, x_stride, y, y_stride, output, output_stride,
         count);
}
} // namespace nncase::ntt