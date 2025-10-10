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

template <bool Arch, size_t in_offset_scale, size_t out_offset_scale, class T1,
          class T2, template <class> class TPostOps, class Stride>
struct u_cast {
  public:
    using T2Elem = element_or_scalar_t<T2>;

    constexpr void operator()(const T1 *input, Stride input_stride, T2 *output,
                              Stride output_stride, size_t count) noexcept {
        using policy_t = u_cast_policy<Arch>;
        constexpr auto unroll = policy_t::unroll;

        if constexpr (in_offset_scale > 1 && out_offset_scale == 1) {
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    prepend_lanes_t<T1, in_offset_scale> in_temp{};
                    auto in_ptr = input;
                    ntt::loop<in_offset_scale>([&](auto s) {
                        in_temp(s) = *(in_ptr + s * input_stride);
                    });
                    *output = ntt::cast_elem<T2Elem>(in_temp);
                    (*output) = TPostOps<T2>()(*output);
                    output += 1;
                    input += ntt::where(input_stride == 1, in_offset_scale, 1);
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                prepend_lanes_t<T1, in_offset_scale> in_temp{};
                auto in_ptr = input;
                ntt::loop<in_offset_scale>(
                    [&](auto s) { in_temp(s) = *(in_ptr + s * input_stride); });
                *output = ntt::cast_elem<T2Elem>(in_temp);
                (*output) = TPostOps<T2>()(*output);
                output += 1;
                input += ntt::where(input_stride == 1, in_offset_scale, 1);
            }

        } else if constexpr (in_offset_scale == 1 && out_offset_scale > 1) {
            using value_type = typename T2::element_type;

            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    auto tmp_output = ntt::cast_elem<T2Elem>(*input);
                    auto out_ptr = output;
                    ntt::loop<out_offset_scale>([&](auto s) {
                        *out_ptr = tmp_output(s);
                        (*out_ptr) = TPostOps<T2>()(*out_ptr);
                        out_ptr += output_stride;
                    });
                    output +=
                        ntt::where(output_stride == 1, out_offset_scale, 1);
                    input += 1;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                auto tmp_output = ntt::cast_elem<T2Elem>(*input);
                auto out_ptr = output;
                ntt::loop<out_offset_scale>([&](auto s) {
                    *out_ptr = tmp_output(s);
                    (*out_ptr) = TPostOps<T2>()(*out_ptr);
                    out_ptr += output_stride;
                });
                output += ntt::where(output_stride == 1, out_offset_scale, 1);
                input += 1;
            }

        } else {
            while (count / unroll) {
                for (size_t i = 0; i < unroll; i++) {
                    *output = ntt::cast_elem<T2Elem>(*input);
                    (*output) = TPostOps<T2>()(*output);
                    input += input_stride * in_offset_scale;
                    output += output_stride * out_offset_scale;
                    count--;
                }
            }

            for (size_t i = 0; i < count; i++) {
                *output = ntt::cast_elem<T2Elem>(*input);
                (*output) = TPostOps<T2>()(*output);
                input += input_stride * in_offset_scale;
                output += output_stride * out_offset_scale;
            }
        }
    }
};
} // namespace ukernels

template <size_t in_offset_scale, size_t out_offset_scale,
          template <class> class TPostOp = DefaultPostOp, class T1, class T2,
          class Stride>
constexpr void u_cast(const T1 *input, Stride input_stride, T2 *output,
                      Stride output_stride, size_t count) noexcept {
    ukernels::u_cast<true, in_offset_scale, out_offset_scale, T1, T2, TPostOp,
                     Stride>
        impl;
    impl(input, input_stride, output, output_stride, count);
}
} // namespace nncase::ntt
