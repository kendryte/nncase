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
#include "../../../primitive_ops.h"
#include "../../../ukernels/u_rope.h"
#include "../arch_types.h"
#include "nncase/ntt/compiler_defs.h"
#include "nncase/ntt/shape.h"
#include <riscv_vector.h>

namespace nncase::ntt::ukernels {
template <size_t NumHeads, size_t HalfDim>
struct u_rope<vector<half, NTT_VLEN / 16>, NumHeads, HalfDim, true> {
  public:
    using T = vector<half, NTT_VLEN / 16>;

    template <Dimension TSeqLen, Strides TInputStrides, Strides TCosStrides,
              Strides TSinStrides, Strides TOutputStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TSeqLen &seq_len, const TInputStrides &input_strides,
               const TCosStrides &cos_strides, const TSinStrides &sin_strides,
               const TOutputStrides &output_strides) noexcept {
        using rope_layout = ukernels::rope_layout;

        constexpr auto unroll = 4_dim;
        ntt::apply_tiled(
            ntt::make_shape(fixed_dim_v<HalfDim>, seq_len),
            ntt::make_shape(1_dim, unroll),
            [&](auto index, auto in_offset, auto cos_offset, auto sin_offset,
                auto out_offset) {
                const auto seq_tile = ntt::min(unroll, seq_len - index[1_dim]);
                asm volatile(
                    "vsetvli zero, %[vl], e16, m4, ta, ma\n" ::[vl] "r"(
                        seq_tile * T::size()));
                const T *NTT_RESTRICT cos_0p = cos + cos_offset;
                const T *NTT_RESTRICT sin_0p = sin + sin_offset;
                const T *NTT_RESTRICT cos_1p =
                    cos_0p +
                    HalfDim * cos_strides[rope_layout::sincos_dim_axis];
                const T *NTT_RESTRICT sin_1p =
                    sin_0p +
                    HalfDim * sin_strides[rope_layout::sincos_dim_axis];

                // v0: cos_0
                // v4: sin_0
                // v8: cos_1
                // v12: sin_1
                // v16: input_4_0
                // v20: input_4_1
                // v24: tmp_0
                // v28: tmp_1
                asm volatile("vle16.v v0, (%[cos_0p])\n"
                             :
                             : [cos_0p] "r"(cos_0p)
                             : "v0", "memory");
                asm volatile("vle16.v v4, (%[sin_0p])\n"
                             :
                             : [sin_0p] "r"(sin_0p)
                             : "v4", "memory");
                asm volatile("vle16.v v8, (%[cos_1p])\n"
                             :
                             : [cos_1p] "r"(cos_1p)
                             : "v8", "memory");
                asm volatile("vle16.v v12, (%[sin_1p])\n"
                             :
                             : [sin_1p] "r"(sin_1p)
                             : "v12", "memory");

                for (size_t h = 0; h < NumHeads; h++) {
                    const T *NTT_RESTRICT input_0p =
                        input + in_offset +
                        h * input_strides[rope_layout::head_axis];
                    const T *NTT_RESTRICT input_1p =
                        input_0p +
                        HalfDim * input_strides[rope_layout::dim_axis];
                    T *NTT_RESTRICT output_0p =
                        output + out_offset +
                        h * output_strides[rope_layout::head_axis];
                    T *NTT_RESTRICT output_1p =
                        output_0p +
                        HalfDim * output_strides[rope_layout::dim_axis];

                    asm volatile("vle16.v v16, (%[input_0p])\n"
                                 :
                                 : [input_0p] "r"(input_0p)
                                 : "v16", "memory");
                    asm volatile("vle16.v v20, (%[input_1p])\n"
                                 :
                                 : [input_1p] "r"(input_1p)
                                 : "v20", "memory");

                    // 2nd half: output_dp[h + output_dim_strides * HalfDim]
                    // = ntt::mul_add(input_1, cos_1, input_0 * sin_1)
                    asm volatile(
                        "vfmul.vv v28, v16, v12\n" // tmp_1 = input_0 * sin_1
                        ::
                            : "v12", "v16", "v28", "memory");
                    asm volatile("vfmacc.vv v28, v20, v8\n" // tmp_1 = input_1 *
                                                            // cos_1 + tmp_1
                                 ::
                                     : "v8", "v20", "v28", "memory");
                    asm volatile("vse16.v v28, (%[output_1p])\n"
                                 :
                                 : [output_1p] "r"(output_1p)
                                 : "v28", "memory");

                    // 1st half: output_dp[h] = ntt::mul_sub(input_0, cos_0,
                    // input_1 * sin_0)
                    asm volatile(
                        "vfmul.vv v24, v20, v4\n" // tmp_0 = input_1 * sin_0
                        ::
                            : "v4", "v20", "v24", "memory");
                    asm volatile("vfmsac.vv v24, v16, v0\n" // tmp_0 = input_0 *
                                                            // cos_0 - tmp_0
                                 ::
                                     : "v0", "v16", "v24", "memory");
                    asm volatile("vse16.v v24, (%[output_0p])\n"
                                 :
                                 : [output_0p] "r"(output_0p)
                                 : "v24", "memory");
                }
            },
            input_strides.template slice<1>(), cos_strides, sin_strides,
            output_strides.template slice<1>());
    }
};
} // namespace nncase::ntt::ukernels
