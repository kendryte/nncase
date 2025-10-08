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
#include "../apply.h"
#include "../dimension.h"
#include "../primitive_ops.h"
#include "nncase/ntt/tensor_traits.h"

namespace nncase::ntt {
namespace ukernels {
struct rope_layout {
    // [head, dim, seq]
    static constexpr auto head_axis = 0_dim;
    static constexpr auto dim_axis = 1_dim;
    static constexpr auto seq_axis = 2_dim;

    static constexpr auto sincos_dim_axis = 0_dim;
    static constexpr auto sincos_seq_axis = 1_dim;
};

template <ScalarOrVector T, size_t NumHeads, size_t HalfDim, bool Arch>
struct u_rope {
  public:
    template <Dimension TSeqLen, Strides TInputStrides, Strides TCosStrides,
              Strides TSinStrides, Strides TOutputStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TSeqLen &seq_len, const TInputStrides &input_strides,
               const TCosStrides &cos_strides, const TSinStrides &sin_strides,
               const TOutputStrides &output_strides) noexcept {
        using rope_layout = ukernels::rope_layout;

        ntt::apply(
            ntt::make_shape(fixed_dim_v<HalfDim>, seq_len),
            [&](auto, auto in_offset, auto cos_offset, auto sin_offset,
                auto out_offset) {
                const auto cos_0 = cos[cos_offset];
                const auto sin_0 = sin[sin_offset];
                const auto cos_1 =
                    cos[cos_offset +
                        HalfDim * cos_strides[rope_layout::sincos_dim_axis]];
                const auto sin_1 =
                    sin[sin_offset +
                        HalfDim * sin_strides[rope_layout::sincos_dim_axis]];

                for (size_t h = 0; h < NumHeads; h++) {
                    const auto input_0 =
                        input[in_offset +
                              h * input_strides[rope_layout::head_axis]];
                    const auto input_1 =
                        input[in_offset +
                              h * input_strides[rope_layout::head_axis] +
                              HalfDim * input_strides[rope_layout::dim_axis]];

                    // 1st half
                    output[out_offset +
                           h * output_strides[rope_layout::head_axis]] =
                        ntt::mul_sub(input_0, cos_0, input_1 * sin_0);

                    // 2nd half
                    output[out_offset +
                           h * output_strides[rope_layout::head_axis] +
                           HalfDim * output_strides[rope_layout::dim_axis]] =
                        ntt::mul_add(input_1, cos_1, input_0 * sin_1);
                }
            },
            input_strides.template slice<1>(), cos_strides, sin_strides,
            output_strides.template slice<1>());
    }
};
} // namespace ukernels

template <size_t NumHeads, size_t HalfDim, ScalarOrVector T, Dimension TSeqLen,
          Strides TInputStrides, Strides TCosStrides, Strides TSinStrides,
          Strides TOutputStrides>
constexpr void
u_rope(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
       const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
       const TSeqLen &seq_len, const TInputStrides &input_strides,
       const TCosStrides &cos_strides, const TSinStrides &sin_strides,
       const TOutputStrides &output_strides) noexcept {
    ukernels::u_rope<T, NumHeads, HalfDim, true> impl;
    impl(input, cos, sin, output, seq_len, input_strides, cos_strides,
         sin_strides, output_strides);
}
} // namespace nncase::ntt
