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

template <size_t NumHeads, size_t HalfDim, bool UseF32>
struct u_rope<vector<half, NTT_VLEN / 16>, NumHeads, HalfDim, true, UseF32> {
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

        constexpr auto unroll = 2_dim;
        ntt::apply_tiled(
            ntt::make_shape(fixed_dim_v<HalfDim>, seq_len),
            ntt::make_shape(1_dim, unroll),
            [&](auto index, auto in_offset, auto cos_offset, auto sin_offset,
                auto out_offset) {
                const auto seq_tile = ntt::min(unroll, seq_len - index[1_dim]);

                size_t vl =
                    __riscv_vsetvl_e16m2((size_t)(seq_tile * T::size()));
                size_t half_vl = vl / 2;

                const T *NTT_RESTRICT cos_0p = cos + cos_offset;
                const T *NTT_RESTRICT sin_0p = sin + sin_offset;
                const T *NTT_RESTRICT cos_1p =
                    cos_0p +
                    HalfDim * cos_strides[rope_layout::sincos_dim_axis];
                const T *NTT_RESTRICT sin_1p =
                    sin_0p +
                    HalfDim * sin_strides[rope_layout::sincos_dim_axis];

                if constexpr (UseF32) {
                    if (unroll == seq_tile) {
                        vfloat16m2_t v0 =
                            __riscv_vle16_v_f16m2((_Float16 *)(cos_0p),
                                                  vl); // cos_0
                        vfloat16m2_t v4 =
                            __riscv_vle16_v_f16m2((_Float16 *)(sin_0p),
                                                  vl); // sin_0
                        vfloat16m2_t v8 =
                            __riscv_vle16_v_f16m2((_Float16 *)(cos_1p),
                                                  vl); // cos_1
                        vfloat16m2_t v12 =
                            __riscv_vle16_v_f16m2((_Float16 *)(sin_1p),
                                                  vl); // sin_1

                        vfloat32m4_t v0_f32 =
                            __riscv_vfwcvt_f_f_v_f32m4(v0, vl);
                        vfloat32m4_t v4_f32 =
                            __riscv_vfwcvt_f_f_v_f32m4(v4, vl);
                        vfloat32m4_t v8_f32 =
                            __riscv_vfwcvt_f_f_v_f32m4(v8, vl);
                        vfloat32m4_t v12_f32 =
                            __riscv_vfwcvt_f_f_v_f32m4(v12, vl);

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

                            vfloat16m2_t v16 = __riscv_vle16_v_f16m2(
                                (const _Float16 *)(input_0p),
                                vl); // input_0
                            vfloat16m2_t v20 = __riscv_vle16_v_f16m2(
                                (const _Float16 *)(input_1p),
                                vl); // input_1

                            vfloat32m4_t v16_f32 = ntt::cast_elem<float>(
                                (ntt::vector<half, NTT_VLEN / 16 * unroll>)v16);
                            vfloat32m4_t v20_f32 = ntt::cast_elem<float>(
                                (ntt::vector<half, NTT_VLEN / 16 * unroll>)v20);

                            prepend_lanes_t<
                                vector<float, NTT_VLEN / 32 * unroll>, 2>
                                v28_f32{};
                            prepend_lanes_t<
                                vector<float, NTT_VLEN / 32 * unroll>, 2>
                                v24_f32{};

                            v28_f32(0_dim) = __riscv_vfmul_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v16_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v12_f32, 0_dim),
                                half_vl);
                            v28_f32(0_dim) = __riscv_vfmacc_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v28_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v20_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v8_f32, 0_dim),
                                half_vl);

                            v24_f32(0_dim) = __riscv_vfmul_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v20_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v4_f32, 0_dim),
                                half_vl);
                            v24_f32(0_dim) = __riscv_vfmsac_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v24_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v16_f32, 0_dim),
                                __riscv_vget_v_f32m4_f32m2(v0_f32, 0_dim),
                                half_vl);

                            v28_f32(1_dim) = __riscv_vfmul_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v16_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v12_f32, 1_dim),
                                half_vl);
                            v28_f32(1_dim) = __riscv_vfmacc_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v28_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v20_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v8_f32, 1_dim),
                                half_vl);

                            v24_f32(1_dim) = __riscv_vfmul_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v20_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v4_f32, 1_dim),
                                half_vl);
                            v24_f32(1_dim) = __riscv_vfmsac_vv_f32m2(
                                __riscv_vget_v_f32m4_f32m2(v24_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v16_f32, 1_dim),
                                __riscv_vget_v_f32m4_f32m2(v0_f32, 1_dim),
                                half_vl);

                            vfloat16m2_t v28 = __riscv_vfncvt_f_f_w_f16m2(
                                (vfloat32m4_t)v28_f32, vl);
                            vfloat16m2_t v24 = __riscv_vfncvt_f_f_w_f16m2(
                                (vfloat32m4_t)v24_f32, vl);

                            __riscv_vse16_v_f16m2((_Float16 *)(output_1p), v28,
                                                  vl);
                            __riscv_vse16_v_f16m2((_Float16 *)(output_0p), v24,
                                                  vl);
                        }
                    } else {
                        vfloat16m1_t v0 =
                            __riscv_vle16_v_f16m1((const _Float16 *)(cos_0p),
                                                  vl); // cos_0
                        vfloat16m1_t v4 =
                            __riscv_vle16_v_f16m1((const _Float16 *)(sin_0p),
                                                  vl); // sin_0
                        vfloat16m1_t v8 =
                            __riscv_vle16_v_f16m1((const _Float16 *)(cos_1p),
                                                  vl); // cos_1
                        vfloat16m1_t v12 =
                            __riscv_vle16_v_f16m1((const _Float16 *)(sin_1p),
                                                  vl); // sin_1

                        vfloat32m2_t v0_f32 =
                            __riscv_vfwcvt_f_f_v_f32m2(v0, vl);
                        vfloat32m2_t v4_f32 =
                            __riscv_vfwcvt_f_f_v_f32m2(v4, vl);
                        vfloat32m2_t v8_f32 =
                            __riscv_vfwcvt_f_f_v_f32m2(v8, vl);
                        vfloat32m2_t v12_f32 =
                            __riscv_vfwcvt_f_f_v_f32m2(v12, vl);

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

                            /* load input halves — f16m1 */
                            vfloat16m1_t v16 = __riscv_vle16_v_f16m1(
                                (const _Float16 *)(input_0p),
                                vl); // input_0
                            vfloat16m1_t v20 = __riscv_vle16_v_f16m1(
                                (const _Float16 *)(input_1p),
                                vl); // input_1

                            vfloat32m2_t v16_f32 = ntt::cast_elem<float>(
                                (ntt::vector<half, NTT_VLEN / 16>)v16);
                            vfloat32m2_t v20_f32 = ntt::cast_elem<float>(
                                (ntt::vector<half, NTT_VLEN / 16>)v20);

                            prepend_lanes_t<vector<float, NTT_VLEN / 32>, 2>
                                v28_f32{};
                            prepend_lanes_t<vector<float, NTT_VLEN / 32>, 2>
                                v24_f32{};

                            v28_f32(0_dim) = __riscv_vfmul_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v16_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v12_f32, 0_dim),
                                half_vl);
                            v28_f32(0_dim) = __riscv_vfmacc_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v28_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v20_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v8_f32, 0_dim),
                                half_vl);

                            v24_f32(0_dim) = __riscv_vfmul_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v20_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v4_f32, 0_dim),
                                half_vl);
                            v24_f32(0_dim) = __riscv_vfmsac_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v24_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v16_f32, 0_dim),
                                __riscv_vget_v_f32m2_f32m1(v0_f32, 0_dim),
                                half_vl);

                            v28_f32(1_dim) = __riscv_vfmul_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v16_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v12_f32, 1_dim),
                                half_vl);
                            v28_f32(1_dim) = __riscv_vfmacc_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v28_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v20_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v8_f32, 1_dim),
                                half_vl);

                            v24_f32(1_dim) = __riscv_vfmul_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v20_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v4_f32, 1_dim),
                                half_vl);
                            v24_f32(1_dim) = __riscv_vfmsac_vv_f32m1(
                                __riscv_vget_v_f32m2_f32m1(v24_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v16_f32, 1_dim),
                                __riscv_vget_v_f32m2_f32m1(v0_f32, 1_dim),
                                half_vl);

                            vfloat16m1_t v28 = __riscv_vfncvt_f_f_w_f16m1(
                                (vfloat32m2_t)v28_f32, vl);
                            vfloat16m1_t v24 = __riscv_vfncvt_f_f_w_f16m1(
                                (vfloat32m2_t)v24_f32, vl);

                            __riscv_vse16_v_f16m1((_Float16 *)(output_1p), v28,
                                                  vl);
                            __riscv_vse16_v_f16m1((_Float16 *)(output_0p), v24,
                                                  vl);
                        }
                    }
                } else {
                    vfloat16m4_t v0 =
                        __riscv_vle16_v_f16m4((const _Float16 *)(cos_0p),
                                              vl); // cos_0
                    vfloat16m4_t v4 =
                        __riscv_vle16_v_f16m4((const _Float16 *)(sin_0p),
                                              vl); // sin_0
                    vfloat16m4_t v8 =
                        __riscv_vle16_v_f16m4((const _Float16 *)(cos_1p),
                                              vl); // cos_1
                    vfloat16m4_t v12 =
                        __riscv_vle16_v_f16m4((const _Float16 *)(sin_1p),
                                              vl); // sin_1

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

                        vfloat16m4_t v16 =
                            __riscv_vle16_v_f16m4((const _Float16 *)(input_0p),
                                                  vl); // input_0
                        vfloat16m4_t v20 =
                            __riscv_vle16_v_f16m4((const _Float16 *)(input_1p),
                                                  vl); // input_1

                        // 2nd half: output_1p = input_1 * cos_1 + input_0 *
                        // sin_1 tmp_1 = input_0 * sin_1
                        vfloat16m4_t v28 = __riscv_vfmul_vv_f16m4(v16, v12, vl);
                        // tmp_1 += input_1 * cos_1
                        v28 = __riscv_vfmacc_vv_f16m4(v28, v20, v8, vl);
                        __riscv_vse16_v_f16m4((_Float16 *)(output_1p), v28, vl);

                        // 1st half: output_0p = input_0 * cos_0 - input_1 *
                        // sin_0 tmp_0 = input_1 * sin_0
                        vfloat16m4_t v24 = __riscv_vfmul_vv_f16m4(v20, v4, vl);
                        v24 = __riscv_vfmsac_vv_f16m4(v24, v16, v0, vl);
                        __riscv_vse16_v_f16m4((_Float16 *)(output_0p), v24, vl);
                    }
                }
            },
            input_strides.template slice<1>(), cos_strides, sin_strides,
            output_strides.template slice<1>());
    }
};

template <size_t NumHeads, size_t HalfDim, bool UseF32>
struct u_rope<vector<float, NTT_VLEN / 32>, NumHeads, HalfDim, true, UseF32> {
  public:
    using T = vector<float, NTT_VLEN / 32>;

    template <Dimension TSeqLen, Strides TInputStrides, Strides TCosStrides,
              Strides TSinStrides, Strides TOutputStrides>
    constexpr void
    operator()(const T *NTT_RESTRICT input, const T *NTT_RESTRICT cos,
               const T *NTT_RESTRICT sin, T *NTT_RESTRICT output,
               const TSeqLen &seq_len, const TInputStrides &input_strides,
               const TCosStrides &cos_strides, const TSinStrides &sin_strides,
               const TOutputStrides &output_strides) noexcept {
        using rope_layout = ukernels::rope_layout;

        constexpr auto unroll = 2_dim;
        ntt::apply_tiled(
            ntt::make_shape(fixed_dim_v<HalfDim>, seq_len),
            ntt::make_shape(1_dim, unroll),
            [&](auto index, auto in_offset, auto cos_offset, auto sin_offset,
                auto out_offset) {
                const auto seq_tile = ntt::min(unroll, seq_len - index[1_dim]);

                size_t vl =
                    __riscv_vsetvl_e32m2((size_t)(seq_tile * T::size()));

                const T *NTT_RESTRICT cos_0p = cos + cos_offset;
                const T *NTT_RESTRICT sin_0p = sin + sin_offset;
                const T *NTT_RESTRICT cos_1p =
                    cos_0p +
                    HalfDim * cos_strides[rope_layout::sincos_dim_axis];
                const T *NTT_RESTRICT sin_1p =
                    sin_0p +
                    HalfDim * sin_strides[rope_layout::sincos_dim_axis];

                {
                    vfloat32m2_t v0 =
                        __riscv_vle32_v_f32m2((const float *)(cos_0p),
                                              vl); // cos_0
                    vfloat32m2_t v4 =
                        __riscv_vle32_v_f32m2((const float *)(sin_0p),
                                              vl); // sin_0
                    vfloat32m2_t v8 =
                        __riscv_vle32_v_f32m2((const float *)(cos_1p),
                                              vl); // cos_1
                    vfloat32m2_t v12 =
                        __riscv_vle32_v_f32m2((const float *)(sin_1p),
                                              vl); // sin_1

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

                        vfloat32m2_t v32 =
                            __riscv_vle32_v_f32m2((const float *)(input_0p),
                                                  vl); // input_0
                        vfloat32m2_t v20 =
                            __riscv_vle32_v_f32m2((const float *)(input_1p),
                                                  vl); // input_1

                        // 2nd half: output_1p = input_1 * cos_1 + input_0 *
                        // sin_1 tmp_1 = input_0 * sin_1
                        vfloat32m2_t v28 = __riscv_vfmul_vv_f32m2(v32, v12, vl);
                        // tmp_1 += input_1 * cos_1
                        v28 = __riscv_vfmacc_vv_f32m2(v28, v20, v8, vl);
                        __riscv_vse32_v_f32m2((float *)(output_1p), v28, vl);

                        // 1st half: output_0p = input_0 * cos_0 - input_1 *
                        // sin_0 tmp_0 = input_1 * sin_0
                        vfloat32m2_t v24 = __riscv_vfmul_vv_f32m2(v20, v4, vl);
                        v24 = __riscv_vfmsac_vv_f32m2(v24, v32, v0, vl);
                        __riscv_vse32_v_f32m2((float *)(output_0p), v24, vl);
                    }
                }
            },
            input_strides.template slice<1>(), cos_strides, sin_strides,
            output_strides.template slice<1>());
    }
};
} // namespace nncase::ntt::ukernels
