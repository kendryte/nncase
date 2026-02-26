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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#if __riscv_vector
#include "utils.h"
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
#if __riscv_vector

static __inline __attribute__((__always_inline__)) vfloat32m8_t
exp_ps2(vfloat32m8_t _p, size_t vl) {
    _p = vfmax_vf_f32m8(_p, -88.0f, vl);
    _p = vfmul_vf_f32m8(_p, 12102203.0f, vl);
    _p = vfadd_vf_f32m8(_p, 1065414017, vl);

    vint32m8_t p2 = vfcvt_x_f_v_i32m8(_p, vl);
    _p = vreinterpret_v_i32m8_f32m8(p2);
    return _p;
}

vfloat32m8_t exp_ps2_opt(vfloat32m8_t _p, const float c0, const float c1,
                         const float c2, size_t vl) {
    _p = vfmax_vf_f32m8(_p, c0, vl);
    _p = vfmadd_vf_f32m8(_p, c1, vfmv_v_f_f32m8(c2, vl), vl);

    vint32m8_t p2 = vfcvt_x_f_v_i32m8(_p, vl);
    _p = vreinterpret_v_i32m8_f32m8(p2);
    return _p;
}

#if 0
result<void> optimized_safe_softmax(const float *input, float *output,
                                    gsl::span<const size_t> in_shape,
                                    int32_t axis, float beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++)
        in_side *= in_shape[i];

    // axis == -1
    if (positive_axis == (ndim - 1)) {
        const float *ptr_input = input;
        float *ptr_output = output;

        for (size_t i = 0; i < out_side; i++) {
            auto n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;

            // max
            float max = *ptr_input_vl;
            {
                size_t vl = vsetvl_e32m4(n);
                vfloat32m4_t s = vfmv_v_f_f32m4(max, vl);
                while (n / vl > 0) {
                    vfloat32m4_t v = vle32_v_f32m4(ptr_input_vl, vl);
                    s = vfmax_vv_f32m4(s, v, vl);

                    n -= vl;
                    ptr_input_vl += vl;
                }

                vfloat32m1_t reduced_max_ = vfredmax_vs_f32m4_f32m1(
                    vundefined_f32m1(), s, vfmv_v_f_f32m1(max, vl), vl);
                max = vfmv_f_s_f32m1_f32(reduced_max_);

                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    s = vfmv_v_f_f32m4(max, vl);
                    vfloat32m4_t v = vle32_v_f32m4(ptr_input_vl, vl);
                    s = vfmax_vv_f32m4(s, v, vl);
                    reduced_max_ = vfredmax_vs_f32m4_f32m1(
                        vundefined_f32m1(), s, vfmv_v_f_f32m1(max, vl), vl);
                    max = vfmv_f_s_f32m1_f32(reduced_max_);
                }
            }

            // exp((x - max) * beta) and sum(exp)
            float sum = 0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            {
                auto vl = vsetvl_e32m4(n);
                auto s = vfmv_v_f_f32m4(0.0f, vl);
                while (n / vl > 0) {

                    auto v_in = vle32_v_f32m4(ptr_input_vl, vl);
                    auto v_out = exp_ps(
                        vfmul_vf_f32m4(vfsub_vf_f32m4(v_in, max, vl), beta, vl),
                        vl);
                    s = vfadd_vv_f32m4(s, v_out, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    n -= vl;
                }
                vfloat32m1_t reduce_sum_ = vfredosum_vs_f32m4_f32m1(
                    vundefined_f32m1(), s, vfmv_v_f_f32m1(0.0f, vl), vl);
                sum += vfmv_f_s_f32m1_f32(reduce_sum_);

                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    auto v_in = vle32_v_f32m4(ptr_input_vl, vl);
                    auto v_out = exp_ps(
                        vfmul_vf_f32m4(vfsub_vf_f32m4(v_in, max, vl), beta, vl),
                        vl);
                    reduce_sum_ =
                        vfredosum_vs_f32m4_f32m1(vundefined_f32m1(), v_out,
                                                 vfmv_v_f_f32m1(0.0f, vl), vl);

                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                    sum += vfmv_f_s_f32m1_f32(reduce_sum_);
                }
            }
            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = 1.0f / sum;
            {
                auto vl = vsetvl_e32m4(n);
                while (n / vl > 0) {
                    auto v_out = vle32_v_f32m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f32m4(v_out, sum, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                    ptr_output_vl += vl;
                    n -= vl;
                }
                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    auto v_out = vle32_v_f32m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f32m4(v_out, sum, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                }
            }

            ptr_input += axis_dim;
            ptr_output += axis_dim;
        }
    } else {
        dims_t axes{positive_axis};
        auto reduced_shape =
            kernels::detail::get_reduced_shape(in_shape, axes, true);
        auto reduced_size = compute_size(reduced_shape);
        std::vector<float> max(reduced_size,
                               std::numeric_limits<float>::lowest());
        std::vector<float> sum(reduced_size, 0.f);

        for (size_t i = 0; i < out_side; i++) {
            const float *ptr_input = input + i * axis_dim * in_side;
            const float *ptr_input_vl = ptr_input;

            float *ptr_output = output + i * axis_dim * in_side;
            float *ptr_output_vl = ptr_output;

            float *ptr_max = max.data() + i * in_side;
            float *ptr_max_vl = ptr_max;

            float *ptr_sum = sum.data() + i * in_side;
            float *ptr_sum_vl = ptr_sum;

            // max
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);

                    v_max = vfmax_vv_f32m8(v_in, v_max, vl);
                    vse32_v_f32m8(ptr_max_vl, v_max, vl);

                    ptr_input_vl += vl;
                    ptr_max_vl += vl;
                    n -= vl;
                }
            }

            // exp((x - max) * beta) and sum(exp)
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    auto v_out =
                        exp_ps(vfmul_vf_f32m8(vfsub_vv_f32m8(v_in, v_max, vl),
                                              beta, vl),
                               vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    v_sum = vfadd_vv_f32m8(v_sum, v_out, vl);
                    vse32_v_f32m8(ptr_sum_vl, v_sum, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    ptr_max_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }

            // div
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_out = vle32_v_f32m8(ptr_output_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    v_out = vfdiv_vv_f32m8(v_out, v_sum, vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    ptr_output_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }
        }
    }
    return ok();
}
#else
#define _RVV_FLOAT32_EXP_OP_OPT(LMUL)                                          \
    static inline vfloat32m##LMUL##_t safe_softmax_exp(vfloat32m##LMUL##_t x,  \
                                                       size_t vl) {            \
        constexpr float c0 = 0x1.ffffe8p-1f;                                   \
        constexpr float c1 = 0x1.fffb34p-1f;                                   \
        constexpr float c2 = 0x1.00059cp-1f;                                   \
        constexpr float c3 = 0x1.57e0b8p-3f;                                   \
        constexpr float c4 = 0x1.53ac16p-5f;                                   \
        x = vfmax_vf_f32m##LMUL(x, c_exp_lo, vl);                              \
        auto b = vfmul_vf_f32m##LMUL(x, c_cephes_LOG2EF, vl);                  \
        auto a = vfcvt_x_f_v_i32m##LMUL(b, vl);                                \
        b = vfcvt_f_x_v_f32m##LMUL(a, vl);                                     \
        x = vfnmsac_vf_f32m##LMUL(x, c_cephes_exp_C1, b, vl);                  \
        x = vfnmsac_vf_f32m##LMUL(x, c_cephes_exp_C2, b, vl);                  \
        b = vfmv_v_f_f32m##LMUL(c3, vl);                                       \
        a = vadd_vx_i32m##LMUL(a, 0x7f, vl);                                   \
        b = vfmacc_vf_f32m##LMUL(b, c4, x, vl);                                \
        auto tmp = vfmv_v_f_f32m##LMUL(c2, vl);                                \
        b = vfmadd_vv_f32m##LMUL(b, x, tmp, vl);                               \
        tmp = vfmv_v_f_f32m##LMUL(c1, vl);                                     \
        b = vfmadd_vv_f32m##LMUL(b, x, tmp, vl);                               \
        tmp = vfmv_v_f_f32m##LMUL(c0, vl);                                     \
        b = vfmadd_vv_f32m##LMUL(b, x, tmp, vl);                               \
        a = vsll_vx_i32m##LMUL(a, 23, vl);                                     \
        auto pow2n = vreinterpret_v_i32m##LMUL##_f32m##LMUL(a);                \
        return vfmul_vv_f32m##LMUL(b, pow2n, vl);                              \
    }

_RVV_FLOAT32_EXP_OP_OPT(1)
_RVV_FLOAT32_EXP_OP_OPT(2)
_RVV_FLOAT32_EXP_OP_OPT(4)
_RVV_FLOAT32_EXP_OP_OPT(8)

result<void> optimized_safe_softmax(const float *input, float *output,
                                    gsl::span<const size_t> in_shape,
                                    int32_t axis, float beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];
    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    if (positive_axis == (ndim - 1)) {
        const size_t vl_max = vsetvlmax_e32m8();
        for (size_t i = 0; i < out_side; i++) {
            const float *ptr_in_base = (const float *)input + i * axis_dim;
            float *ptr_out_base = (float *)output + i * axis_dim;

            // -----------------------------------------------------------
            // Pass 1: Global Max
            // -----------------------------------------------------------
            float max_val = -std::numeric_limits<float>::infinity();
            auto v_max_acc = vfmv_v_f_f32m8(max_val, vl_max);

            size_t n = axis_dim;
            const float *p = ptr_in_base;

            for (size_t vl; n > 0; n -= vl, p += vl) {
                vl = vsetvl_e32m8(n);
                auto v0 = vle32_v_f32m8(p, vl);
                v_max_acc = vfmax_vv_f32m8(v_max_acc, v0, vl);
            }

            auto v_scalar_max = vfmv_s_f_f32m1(vundefined_f32m1(), max_val, 1);
            v_scalar_max = vfredmax_vs_f32m8_f32m1(v_scalar_max, v_max_acc,
                                                   v_scalar_max, vl_max);
            max_val = vfmv_f_s_f32m1_f32(v_scalar_max);

            // -----------------------------------------------------------
            // Pass 2: Exp & Global Sum
            // -----------------------------------------------------------
            auto v_sum_acc = vfmv_v_f_f32m8(0.0f, vl_max);

            n = axis_dim;
            p = ptr_in_base;
            float *pout = ptr_out_base;

            for (size_t vl; n > 0; n -= vl, p += vl, pout += vl) {
                vl = vsetvl_e32m8(n);
                auto v0 = vle32_v_f32m8(p, vl);
                v0 = vfsub_vf_f32m8(v0, max_val, vl);
                if (beta != 1.0f)
                    v0 = vfmul_vf_f32m8(v0, beta, vl);
                v0 = safe_softmax_exp(v0, vl);
                v_sum_acc = vfadd_vv_f32m8(v_sum_acc, v0, vl);
                vse32_v_f32m8(pout, v0, vl);
            }

            auto v_scalar_sum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.0f, 1);
            v_scalar_sum = vfredusum_vs_f32m8_f32m1(v_scalar_sum, v_sum_acc,
                                                    v_scalar_sum, vl_max);
            float sum_val = vfmv_f_s_f32m1_f32(v_scalar_sum);

            // -----------------------------------------------------------
            // Pass 3: Normalize
            // -----------------------------------------------------------
            float inv_sum = 1.0f / sum_val;
            n = axis_dim;
            pout = ptr_out_base;
            for (size_t vl; n > 0; n -= vl, pout += vl) {
                vl = vsetvl_e32m8(n);
                auto v0 = vle32_v_f32m8(pout, vl);
                v0 = vfmul_vf_f32m8(v0, inv_sum, vl);
                vse32_v_f32m8(pout, v0, vl);
            }
        }
    }
    // =========================================================
    // 分支 2: Axis 不是最后一维 (Strided Access) -> 使用 Tiling 优化
    // =========================================================
    else {
        size_t in_side = 1;
        for (size_t i = positive_axis + 1; i < ndim; i++)
            in_side *= in_shape[i];

        constexpr int TILE_SIZE = 1024;
        float local_max[TILE_SIZE];
        float local_sum[TILE_SIZE];
        for (size_t i = 0; i < out_side; i++) {
            const float *batch_in =
                (const float *)input + i * axis_dim * in_side;
            float *batch_out = (float *)output + i * axis_dim * in_side;
            for (size_t col_base = 0; col_base < in_side;
                 col_base += TILE_SIZE) {
                size_t current_tile_w =
                    std::min((size_t)TILE_SIZE, in_side - col_base);

                // --- Pass 1: 计算 Max (在当前 Tile 内) ---
                {
                    const float *ptr_row = batch_in + col_base;
                    float *ptr_max = local_max;
                    size_t n = current_tile_w;
                    while (n > 0) {
                        size_t vl = vsetvl_e32m8(n);
                        auto v_in = vle32_v_f32m8(ptr_row, vl);
                        vse32_v_f32m8(ptr_max, v_in, vl);
                        ptr_row += vl;
                        ptr_max += vl;
                        n -= vl;
                    }

                    ptr_row = batch_in + col_base + in_side;
                    for (size_t r = 1; r < axis_dim; ++r) {
                        n = current_tile_w;
                        ptr_max = local_max;
                        const float *ptr_curr = ptr_row;
                        while (n > 0) {
                            size_t vl = vsetvl_e32m8(n);
                            auto v_in = vle32_v_f32m8(ptr_curr, vl);
                            auto v_max = vle32_v_f32m8(ptr_max, vl);
                            v_max = vfmax_vv_f32m8(v_max, v_in, vl);
                            vse32_v_f32m8(ptr_max, v_max, vl);
                            ptr_curr += vl;
                            ptr_max += vl;
                            n -= vl;
                        }
                        ptr_row += in_side;
                    }
                }

                // --- Pass 2: 计算 Exp 和 Sum (在当前 Tile 内) ---
                {
                    std::fill(local_sum, local_sum + current_tile_w, 0.0f);

                    const float *ptr_in_row = batch_in + col_base;
                    float *ptr_out_row = batch_out + col_base;

                    for (size_t r = 0; r < axis_dim; ++r) {
                        size_t n = current_tile_w;
                        float *ptr_max = local_max;
                        float *ptr_sum = local_sum;
                        const float *ptr_curr = ptr_in_row;
                        float *ptr_out = ptr_out_row;

                        while (n > 0) {
                            size_t vl = vsetvl_e32m8(n);
                            auto v_in = vle32_v_f32m8(ptr_curr, vl);
                            auto v_max = vle32_v_f32m8(ptr_max, vl);

                            // exp((x - max) * beta)
                            auto v_val = vfsub_vv_f32m8(v_in, v_max, vl);
                            if (beta != 1.0f)
                                v_val = vfmul_vf_f32m8(v_val, beta, vl);
                            v_val = safe_softmax_exp(v_val, vl);

                            // 累加 Sum
                            auto v_s = vle32_v_f32m8(ptr_sum, vl);
                            vse32_v_f32m8(ptr_out, v_val, vl);
                            v_s = vfadd_vv_f32m8(v_s, v_val, vl);
                            vse32_v_f32m8(ptr_sum, v_s, vl);

                            ptr_curr += vl;
                            ptr_max += vl;
                            ptr_sum += vl;
                            ptr_out += vl;
                            n -= vl;
                        }
                        ptr_in_row += in_side;
                        ptr_out_row += in_side;
                    }
                }

                // --- Pass 3: 归一化 (在当前 Tile 内) ---
                {
                    // 3.1 预先计算 inv_sum
                    size_t n = current_tile_w;
                    float *ptr_sum = local_sum;
                    while (n > 0) {
                        size_t vl = vsetvl_e32m8(n);
                        auto v_s = vle32_v_f32m8(ptr_sum, vl);
                        v_s = vfrdiv_vf_f32m8(v_s, 1.0f, vl);
                        vse32_v_f32m8(ptr_sum, v_s, vl);
                        ptr_sum += vl;
                        n -= vl;
                    }

                    // 3.2 遍历所有行进行乘法
                    float *ptr_out_row = batch_out + col_base;
                    for (size_t r = 0; r < axis_dim; ++r) {
                        n = current_tile_w;
                        ptr_sum = local_sum;
                        float *ptr_out = ptr_out_row;

                        while (n > 0) {
                            size_t vl = vsetvl_e32m8(n);
                            auto v_val = vle32_v_f32m8(ptr_out, vl);
                            auto v_inv = vle32_v_f32m8(ptr_sum, vl);
                            v_val = vfmul_vv_f32m8(v_val, v_inv, vl);
                            vse32_v_f32m8(ptr_out, v_val, vl);
                            ptr_out += vl;
                            ptr_sum += vl;
                            n -= vl;
                        }
                        ptr_out_row += in_side;
                    }
                }

            } // End of Tile Loop
        }     // End of Batch Loop
    }

    return ok();
}
#endif

#if 0
result<void> optimized_safe_softmax(const __float16_t *input, __float16_t *output,
                                         gsl::span<const size_t> in_shape,
                                         int32_t axis,
                                         __float16_t beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++)
        in_side *= in_shape[i];

    // axis == -1
    if (positive_axis == (ndim - 1)) {
        const __float16_t *ptr_input = input;
        __float16_t *ptr_output = output;

        for (size_t i = 0; i < out_side; i++) {
            auto n = axis_dim;
            const __float16_t *ptr_input_vl = ptr_input;
            __float16_t *ptr_output_vl = ptr_output;

            // max
            __float16_t max = *ptr_input_vl;
            {
                size_t vl = vsetvl_e16m4(n);
                vfloat16m4_t v_max = vfmv_v_f_f16m4(max, vl);
                while (n / vl > 0) {
                    vfloat16m4_t v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    v_max = vfmax_vv_f16m4(v_max, v_in, vl);

                    n -= vl;
                    ptr_input_vl += vl;
                }
                vfloat16m1_t reduced_max_ = vfredmax_vs_f16m4_f16m1(
                    vundefined_f16m1(), v_max, vfmv_v_f_f16m1(max, vl), vl);
                max = vfmv_f_s_f16m1_f16(reduced_max_);

                if (n > 0) {
                    vl = vsetvl_e16m4(n);
                    v_max = vfmv_v_f_f16m4(max, vl);
                    vfloat16m4_t v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    v_max = vfmax_vv_f16m4(v_max, v_in, vl);
                    reduced_max_ = vfredmax_vs_f16m4_f16m1(
                        vundefined_f16m1(), v_max, vfmv_v_f_f16m1(max, vl), vl);
                    max = vfmv_f_s_f16m1_f16(reduced_max_);
                }
            }

            // exp((x - max) * beta) and sum(exp)
            __float16_t sum = (__float16_t)0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            {
                auto vl = vsetvl_e16m4(n);
                auto v_sum = vfmv_v_f_f16m4((__float16_t)0.0f, vl);

                const __float16_t exp_clamp_min = (__float16_t)(-10.0f);
                const __float16_t exp_clamp_max = (__float16_t)(10.0f);

                while (n / vl > 0) {
                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_sub = vfsub_vf_f16m4(v_in, max, vl);
                    auto v_scaled = vfmul_vf_f16m4(v_sub, beta, vl);

                    v_scaled = vfmax_vf_f16m4(v_scaled, exp_clamp_min, vl);
                    v_scaled = vfmin_vf_f16m4(v_scaled, exp_clamp_max, vl);

                    auto v_out = exp_ph(v_scaled, vl);
                    v_sum = vfadd_vv_f16m4(v_sum, v_out, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    n -= vl;
                }
                vfloat16m1_t reduced_sum_ = vfredosum_vs_f16m4_f16m1(
                    vundefined_f16m1(), v_sum,
                    vfmv_v_f_f16m1((__float16_t)0.0f, vl), vl);
                sum += vfmv_f_s_f16m1_f16(reduced_sum_);

                if (n > 0) {
                    vl = vsetvl_e16m4(n);
                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_sub = vfsub_vf_f16m4(v_in, max, vl);
                    auto v_scaled = vfmul_vf_f16m4(v_sub, beta, vl);

                    v_scaled = vfmax_vf_f16m4(v_scaled, exp_clamp_min, vl);
                    v_scaled = vfmin_vf_f16m4(v_scaled, exp_clamp_max, vl);

                    auto v_out = exp_ph(v_scaled, vl);
                    reduced_sum_ = vfredosum_vs_f16m4_f16m1(
                        vundefined_f16m1(), v_out,
                        vfmv_v_f_f16m1((__float16_t)0.0f, vl), vl);

                    vse16_v_f16m4(ptr_output_vl, v_out, vl);
                    sum += vfmv_f_s_f16m1_f16(reduced_sum_);
                }
            }

            if (sum <= (__float16_t)0.0f || !std::isfinite((float)sum)) {
                __float16_t uniform_prob = (__float16_t)(1.0f / axis_dim);
                for (size_t j = 0; j < axis_dim; j++) {
                    ptr_output[j] = uniform_prob;
                }

                ptr_input += axis_dim;
                ptr_output += axis_dim;
                continue;
            }

            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = (__float16_t)1.0f / sum;

            {
                auto vl = vsetvl_e16m4(n);
                while (n / vl > 0) {
                    auto v_out = vle16_v_f16m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f16m4(v_out, sum, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

                    ptr_output_vl += vl;
                    n -= vl;
                }
                if (n > 0) {
                    vl = vsetvl_e16m4(n);
                    auto v_out = vle16_v_f16m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f16m4(v_out, sum, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);
                }
            }

            ptr_input += axis_dim;
            ptr_output += axis_dim;
        }
    } else {
        dims_t axes{positive_axis};
        auto reduced_shape =
            kernels::detail::get_reduced_shape(in_shape, axes, true);
        auto reduced_size = compute_size(reduced_shape);
        std::vector<__float16_t> max(
            reduced_size, std::numeric_limits<__float16_t>::lowest());
        std::vector<__float16_t> sum(reduced_size, (__float16_t)0.f);

        for (size_t i = 0; i < out_side; i++) {
            const __float16_t *ptr_input = input + i * axis_dim * in_side;
            const __float16_t *ptr_input_vl = ptr_input;

            __float16_t *ptr_output = output + i * axis_dim * in_side;
            __float16_t *ptr_output_vl = ptr_output;

            __float16_t *ptr_max = max.data() + i * in_side;
            __float16_t *ptr_max_vl = ptr_max;

            __float16_t *ptr_sum = sum.data() + i * in_side;
            __float16_t *ptr_sum_vl = ptr_sum;

            // max
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e16m4(n);

                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_max = vle16_v_f16m4(ptr_max_vl, vl);

                    v_max = vfmax_vv_f16m4(v_in, v_max, vl);
                    vse16_v_f16m4(ptr_max_vl, v_max, vl);

                    ptr_input_vl += vl;
                    ptr_max_vl += vl;
                    n -= vl;
                }
            }

            // exp((x - max) * beta) and sum(exp)
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e16m4(n);

                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_max = vle16_v_f16m4(ptr_max_vl, vl);
                    auto v_sum = vle16_v_f16m4(ptr_sum_vl, vl);

                    // Calculate (x - max) * beta and clamp the range
                    auto v_scaled = vfmul_vf_f16m4(
                        vfsub_vv_f16m4(v_in, v_max, vl), beta, vl);

                    // Float16 exp safe range clamping
                    const __float16_t exp_clamp_min = (__float16_t)(-10.0f);
                    const __float16_t exp_clamp_max = (__float16_t)(10.0f);
                    v_scaled = vfmax_vf_f16m4(v_scaled, exp_clamp_min, vl);
                    v_scaled = vfmin_vf_f16m4(v_scaled, exp_clamp_max, vl);

                    auto v_out = exp_ph(v_scaled, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

                    v_sum = vfadd_vv_f16m4(v_sum, v_out, vl);
                    vse16_v_f16m4(ptr_sum_vl, v_sum, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    ptr_max_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }

            // div
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e16m4(n);

                    auto v_out = vle16_v_f16m4(ptr_output_vl, vl);
                    auto v_sum = vle16_v_f16m4(ptr_sum_vl, vl);

                    v_out = vfdiv_vv_f16m4(v_out, v_sum, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

                    ptr_output_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }
        }
    }
    return ok();
}

#else
#define _RVV_FLOAT16_EXP_OP_OPT(LMUL)                                          \
    static inline vfloat16m##LMUL##_t safe_softmax_exp(vfloat16m##LMUL##_t x,  \
                                                       size_t vl) {            \
        constexpr __float16_t c0 = (__float16_t)0x1.ffffe8p-1f;                \
        constexpr __float16_t c1 = (__float16_t)0x1.fffb34p-1f;                \
        constexpr __float16_t c2 = (__float16_t)0x1.00059cp-1f;                \
        constexpr __float16_t c3 = (__float16_t)0x1.57e0b8p-3f;                \
        constexpr __float16_t c4 = (__float16_t)0x1.53ac16p-5f;                \
        x = vfmax_vf_f16m##LMUL(x, (__float16_t)c_exp_lo_half, vl);            \
        auto b = vfmul_vf_f16m##LMUL(x, (__float16_t)c_cephes_LOG2EF, vl);     \
        auto a = vfcvt_x_f_v_i16m##LMUL(b, vl);                                \
        b = vfcvt_f_x_v_f16m##LMUL(a, vl);                                     \
        x = vfnmsac_vf_f16m##LMUL(x, (__float16_t)c_cephes_exp_C1, b, vl);     \
        x = vfnmsac_vf_f16m##LMUL(x, (__float16_t)c_cephes_exp_C2, b, vl);     \
        b = vfmv_v_f_f16m##LMUL(c3, vl);                                       \
        a = vadd_vx_i16m##LMUL(a, 15, vl);                                     \
        b = vfmacc_vf_f16m##LMUL(b, c4, x, vl);                                \
        auto tmp = vfmv_v_f_f16m##LMUL(c2, vl);                                \
        b = vfmadd_vv_f16m##LMUL(b, x, tmp, vl);                               \
        tmp = vfmv_v_f_f16m##LMUL(c1, vl);                                     \
        b = vfmadd_vv_f16m##LMUL(b, x, tmp, vl);                               \
        tmp = vfmv_v_f_f16m##LMUL(c0, vl);                                     \
        b = vfmadd_vv_f16m##LMUL(b, x, tmp, vl);                               \
        a = vsll_vx_i16m##LMUL(a, 10, vl);                                     \
        auto pow2n = vreinterpret_v_i16m##LMUL##_f16m##LMUL(a);                \
        return vfmul_vv_f16m##LMUL(b, pow2n, vl);                              \
    }

_RVV_FLOAT16_EXP_OP_OPT(1)
_RVV_FLOAT16_EXP_OP_OPT(2)
_RVV_FLOAT16_EXP_OP_OPT(4)
_RVV_FLOAT16_EXP_OP_OPT(8)

result<void> optimized_safe_softmax(const __float16_t *input,
                                    __float16_t *output,
                                    gsl::span<const size_t> in_shape,
                                    int32_t axis, __float16_t beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    if (positive_axis == (ndim - 1)) {
        const size_t vl_max = vsetvlmax_e16m8();
        for (size_t i = 0; i < out_side; i++) {
            const __float16_t *ptr_in_base =
                (const __float16_t *)input + i * axis_dim;
            __float16_t *ptr_out_base = (__float16_t *)output + i * axis_dim;

            // -----------------------------------------------------------
            // Pass 1: Global Max
            // -----------------------------------------------------------
            __float16_t max_val = -std::numeric_limits<__float16_t>::infinity();
            auto v_max_acc = vfmv_v_f_f16m8(max_val, vl_max);

            size_t n = axis_dim;
            const __float16_t *p = ptr_in_base;

            for (size_t vl; n > 0; n -= vl, p += vl) {
                vl = vsetvl_e16m8(n);
                auto v0 = vle16_v_f16m8(p, vl);
                v_max_acc = vfmax_vv_f16m8(v_max_acc, v0, vl);
            }

            auto v_scalar_max = vfmv_s_f_f16m1(vundefined_f16m1(), max_val, 1);
            v_scalar_max = vfredmax_vs_f16m8_f16m1(v_scalar_max, v_max_acc,
                                                   v_scalar_max, vl_max);
            max_val = vfmv_f_s_f16m1_f16(v_scalar_max);

            // -----------------------------------------------------------
            // Pass 2: Exp & Global Sum
            // -----------------------------------------------------------
            auto v_sum_acc = vfmv_v_f_f16m8((__float16_t)0.0f, vl_max);

            n = axis_dim;
            p = ptr_in_base;
            __float16_t *pout = ptr_out_base;

            for (size_t vl; n > 0; n -= vl, p += vl, pout += vl) {
                vl = vsetvl_e16m8(n);
                auto v0 = vle16_v_f16m8(p, vl);
                v0 = vfsub_vf_f16m8(v0, max_val, vl);
                if (beta != (__float16_t)1.0f)
                    v0 = vfmul_vf_f16m8(v0, beta, vl);
                v0 = safe_softmax_exp(v0, vl);
                v_sum_acc = vfadd_vv_f16m8(v_sum_acc, v0, vl);
                vse16_v_f16m8(pout, v0, vl);
            }

            auto v_scalar_sum =
                vfmv_s_f_f16m1(vundefined_f16m1(), (__float16_t)0.0f, 1);
            v_scalar_sum = vfredusum_vs_f16m8_f16m1(v_scalar_sum, v_sum_acc,
                                                    v_scalar_sum, vl_max);
            auto sum_val = vfmv_f_s_f16m1_f16(v_scalar_sum);

            // -----------------------------------------------------------
            // Pass 3: Normalize
            // -----------------------------------------------------------
            auto inv_sum = (__float16_t)1.0f / sum_val;

            n = axis_dim;
            pout = ptr_out_base;
            for (size_t vl; n > 0; n -= vl, pout += vl) {
                vl = vsetvl_e16m8(n);
                auto v0 = vle16_v_f16m8(pout, vl);
                v0 = vfmul_vf_f16m8(v0, inv_sum, vl);
                vse16_v_f16m8(pout, v0, vl);
            }
        }
    }
    // =========================================================
    // 分支 2: Axis 不是最后一维 (Strided Access) -> 使用 Tiling 优化
    // =========================================================
    else {
        size_t in_side = 1;
        for (size_t i = positive_axis + 1; i < ndim; i++)
            in_side *= in_shape[i];

        constexpr int TILE_SIZE = 2048;
        __float16_t local_max[TILE_SIZE];
        __float16_t local_sum[TILE_SIZE];
        for (size_t i = 0; i < out_side; i++) {
            const __float16_t *batch_in =
                (const __float16_t *)input + i * axis_dim * in_side;
            __float16_t *batch_out =
                (__float16_t *)output + i * axis_dim * in_side;
            for (size_t col_base = 0; col_base < in_side;
                 col_base += TILE_SIZE) {
                size_t current_tile_w =
                    std::min((size_t)TILE_SIZE, in_side - col_base);

                // --- Pass 1: 计算 Max (在当前 Tile 内) ---
                {
                    const __float16_t *ptr_row = batch_in + col_base;
                    __float16_t *ptr_max = local_max;
                    size_t n = current_tile_w;
                    while (n > 0) {
                        size_t vl = vsetvl_e16m8(n);
                        auto v_in = vle16_v_f16m8(ptr_row, vl);
                        vse16_v_f16m8(ptr_max, v_in, vl);
                        ptr_row += vl;
                        ptr_max += vl;
                        n -= vl;
                    }

                    ptr_row = batch_in + col_base + in_side;
                    for (size_t r = 1; r < axis_dim; ++r) {
                        n = current_tile_w;
                        ptr_max = local_max;
                        const __float16_t *ptr_curr = ptr_row;
                        while (n > 0) {
                            size_t vl = vsetvl_e16m8(n);
                            auto v_in = vle16_v_f16m8(ptr_curr, vl);
                            auto v_max = vle16_v_f16m8(ptr_max, vl);
                            v_max = vfmax_vv_f16m8(v_max, v_in, vl);
                            vse16_v_f16m8(ptr_max, v_max, vl);
                            ptr_curr += vl;
                            ptr_max += vl;
                            n -= vl;
                        }
                        ptr_row += in_side;
                    }
                }

                // --- Pass 2: 计算 Exp 和 Sum (在当前 Tile 内) ---
                {
                    std::fill(local_sum, local_sum + current_tile_w, 0.0f);

                    const __float16_t *ptr_in_row = batch_in + col_base;
                    __float16_t *ptr_out_row = batch_out + col_base;

                    for (size_t r = 0; r < axis_dim; ++r) {
                        size_t n = current_tile_w;
                        __float16_t *ptr_max = local_max;
                        __float16_t *ptr_sum = local_sum;
                        const __float16_t *ptr_curr = ptr_in_row;
                        __float16_t *ptr_out = ptr_out_row;

                        while (n > 0) {
                            size_t vl = vsetvl_e16m8(n);
                            auto v_in = vle16_v_f16m8(ptr_curr, vl);
                            auto v_max = vle16_v_f16m8(ptr_max, vl);

                            // exp((x - max) * beta)
                            auto v_val = vfsub_vv_f16m8(v_in, v_max, vl);
                            if (beta != (__float16_t)1.0f)
                                v_val = vfmul_vf_f16m8(v_val, beta, vl);
                            v_val = safe_softmax_exp(v_val, vl);

                            // 累加 Sum
                            auto v_s = vle16_v_f16m8(ptr_sum, vl);
                            vse16_v_f16m8(ptr_out, v_val, vl);
                            v_s = vfadd_vv_f16m8(v_s, v_val, vl);
                            vse16_v_f16m8(ptr_sum, v_s, vl);

                            ptr_curr += vl;
                            ptr_max += vl;
                            ptr_sum += vl;
                            ptr_out += vl;
                            n -= vl;
                        }
                        ptr_in_row += in_side;
                        ptr_out_row += in_side;
                    }
                }

                // --- Pass 3: 归一化 (在当前 Tile 内) ---
                {
                    // 3.1 预先计算 inv_sum
                    size_t n = current_tile_w;
                    __float16_t *ptr_sum = local_sum;
                    while (n > 0) {
                        size_t vl = vsetvl_e16m8(n);
                        auto v_s = vle16_v_f16m8(ptr_sum, vl);
                        v_s = vfrdiv_vf_f16m8(v_s, (__float16_t)1.0f, vl);
                        vse16_v_f16m8(ptr_sum, v_s, vl);
                        ptr_sum += vl;
                        n -= vl;
                    }

                    // 3.2 遍历所有行进行乘法
                    __float16_t *ptr_out_row = batch_out + col_base;
                    for (size_t r = 0; r < axis_dim; ++r) {
                        n = current_tile_w;
                        ptr_sum = local_sum;
                        __float16_t *ptr_out = ptr_out_row;

                        while (n > 0) {
                            size_t vl = vsetvl_e16m8(n);
                            auto v_val = vle16_v_f16m8(ptr_out, vl);
                            auto v_inv = vle16_v_f16m8(ptr_sum, vl);
                            v_val = vfmul_vv_f16m8(v_val, v_inv, vl);
                            vse16_v_f16m8(ptr_out, v_val, vl);
                            ptr_out += vl;
                            ptr_sum += vl;
                            n -= vl;
                        }
                        ptr_out_row += in_side;
                    }
                }

            } // End of Tile Loop
        }     // End of Batch Loop
    }

    return ok();
}
#endif
#endif
} // namespace

#define IN_CAST(_ty, _name) reinterpret_cast<const _ty *>(_name)
#define OUT_CAST(_ty, _name) reinterpret_cast<_ty *>(_name)

// template <typename T>
result<void> optimized::softmax([[maybe_unused]] typecode_t typecode,
                                const gsl::byte *input, gsl::byte *output,
                                gsl::span<const size_t> in_shape,
                                gsl::span<const size_t> in_strides,
                                gsl::span<const size_t> out_strides,
                                int32_t axis, float beta) noexcept {
#if __riscv_vector
    if (typecode == typecode_t::dt_float16) {
        return optimized_safe_softmax(IN_CAST(__float16_t, input),
                                      OUT_CAST(__float16_t, output), in_shape,
                                      axis, __float16_t(beta));
    } else if (typecode == typecode_t::dt_float32) {
        return optimized_safe_softmax(IN_CAST(float, input),
                                      OUT_CAST(float, output), in_shape, axis,
                                      beta);
    } else
#endif
        return stackvm::reference::softmax(typecode, input, output, in_shape,
                                           in_strides, out_strides, axis, beta);
}