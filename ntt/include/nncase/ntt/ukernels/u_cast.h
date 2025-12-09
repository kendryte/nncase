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
#include "../post_ops.h"
#include "../primitive_ops.h"
#include "../vector.h"
#include "nncase/ntt/tensor_traits.h"

namespace nncase::ntt {
namespace ukernels {

template <bool Arch> struct u_cast_policy {
    static constexpr size_t unroll = 2;
};

template <bool Arch, class T1, class T2, template <class> class TPostOps, class Stride>
struct u_cast {
  public:
    using T2Elem = element_or_scalar_t<T2>;

    constexpr void operator()(const T1 *input, Stride input_stride, T2 *output,
                              Stride output_stride, size_t count) noexcept {
        using policy_t = u_cast_policy<Arch>;
        constexpr auto unroll = policy_t::unroll;
        while (count / unroll) {
            for (size_t i = 0; i < unroll; i++) {
                *output = ntt::cast_elem<T2Elem>(*input);
                (*output) = TPostOps<T2>()(*output);
                input += input_stride;
                output += output_stride;
                count--;
            }
        }

        for (size_t i = 0; i < count; i++) {
            *output = ntt::cast_elem<T2Elem>(*input);
            (*output) = TPostOps<T2>()(*output);
            input += input_stride;
            output += output_stride;
        }
    }
};
} // namespace ukernels

template <template <class> class TPostOp = DefaultPostOp, class T1, class T2,
          class Stride>
constexpr void u_cast(const T1 *input, Stride input_stride, T2 *output,
                      Stride output_stride, size_t count) noexcept {
    ukernels::u_cast<true, T1, T2, TPostOp, Stride> impl;
    impl(input, input_stride, output, output_stride, count);
}
} // namespace nncase::ntt
