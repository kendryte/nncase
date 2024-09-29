/* Copyright 2019-2024 Canaan Inc.
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
#include <cmath>

#if __riscv_vector
#include <riscv_vector.h>

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

/*
log(x) = x - x^2 / 2 + p8x^3 + ... + p0x^11 + (q1 + q2) * e
       = x + x^2(-1/2 + p8x + ... +  x^2(p1 + p0x)) + (q1 + q2) * e
*/

#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        auto ux = __riscv_vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);           \
        auto emm0 = __riscv_vsra_vx_i32m##LMUL(ux, 23, vl);                    \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = __riscv_vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);              \
        emm0 = __riscv_vsub_vx_i32m##LMUL(emm0, 0x7e, vl);                     \
        ux = __riscv_vor_vx_i32m##LMUL(                                        \
            ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);              \
        x = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                \
                                                                               \
        auto e = __riscv_vfcvt_f_x_v_f32m##LMUL(emm0, vl);                     \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        auto mask =                                                            \
            __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);     \
        x = __riscv_vfadd_vv_f32m##LMUL##_m(mask, x, x, vl);                   \
        e = __riscv_vfsub_vf_f32m##LMUL##_m(mask, e, 1.f, vl);                 \
        x = __riscv_vfsub_vf_f32m##LMUL(x, 1.f, vl);                           \
                                                                               \
        auto y1 = __riscv_vmv_v_v_f32m##LMUL(x, vl);                           \
        auto y2 = __riscv_vmv_v_v_f32m##LMUL(x, vl);                           \
        auto y3 = __riscv_vmv_v_v_f32m##LMUL(x, vl);                           \
        auto y4 = __riscv_vmv_v_v_f32m##LMUL(x, vl);                           \
        auto y5 = __riscv_vmv_v_v_f32m##LMUL(x, vl);                           \
        auto c1 = __riscv_vfmv_v_f_f32m##LMUL(c_cephes_log_p1, vl);            \
        auto c3 = __riscv_vfmv_v_f_f32m##LMUL(c_cephes_log_p3, vl);            \
        auto c5 = __riscv_vfmv_v_f_f32m##LMUL(c_cephes_log_p5, vl);            \
        auto c7 = __riscv_vfmv_v_f_f32m##LMUL(c_cephes_log_p7, vl);            \
        auto minus_half = __riscv_vfmv_v_f_f32m##LMUL(-0.5f, vl);              \
        auto x2 = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                       \
        y1 = __riscv_vfmadd_vf_f32m##LMUL(y1, c_cephes_log_p0, c1, vl);        \
        y2 = __riscv_vfmadd_vf_f32m##LMUL(y2, c_cephes_log_p2, c3, vl);        \
        y3 = __riscv_vfmadd_vf_f32m##LMUL(y3, c_cephes_log_p4, c5, vl);        \
        y1 = __riscv_vfmadd_vv_f32m##LMUL(y1, x2, y2, vl);                     \
        y4 = __riscv_vfmadd_vf_f32m##LMUL(y4, c_cephes_log_p6, c7, vl);        \
        y1 = __riscv_vfmadd_vv_f32m##LMUL(y1, x2, y3, vl);                     \
        y5 =                                                                   \
            __riscv_vfmadd_vf_f32m##LMUL(y5, c_cephes_log_p8, minus_half, vl); \
        y1 = __riscv_vfmadd_vv_f32m##LMUL(y1, x2, y4, vl);                     \
        y1 = __riscv_vfmadd_vv_f32m##LMUL(y1, x2, y5, vl);                     \
        e = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q1 + c_cephes_log_q2,  \
                                        vl);                                   \
        y1 = __riscv_vfmadd_vv_f32m##LMUL(y1, x2, x, vl);                      \
        return __riscv_vfadd_vv_f32m##LMUL(y1, e, vl);                         \
    }

_RVV_FLOAT32_LOG_OP(1, 32)
_RVV_FLOAT32_LOG_OP(2, 16)
_RVV_FLOAT32_LOG_OP(4, 8)
_RVV_FLOAT32_LOG_OP(8, 4)

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

// e^x = 1 + x + 1/2!x^2 + 1/3!x^3 + 1/4!x^4 + 1/5!x^5 + 1/6!x^6 + 1/7!x^7
#define _RVV_FLOAT_EXP_OP(LMUL, MLEN, TLEN, E, M)                              \
    static inline __attribute__((optimize("no-schedule-insns2")))              \
        vfloat##TLEN##m##LMUL##_t                                              \
        exp_ps(vfloat##TLEN##m##LMUL##_t x, size_t vl) {                       \
        auto a1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_LOG2EF, vl);      \
        auto c1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p1, vl);      \
        auto c3 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p3, vl);      \
        auto c5 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p5, vl);      \
        x = __riscv_vfmin_vf_f##TLEN##m##LMUL(x, c_exp_hi, vl);                \
        x = __riscv_vfmax_vf_f##TLEN##m##LMUL(x, c_exp_lo, vl);                \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        a1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(a1, x, c5, vl);                \
                                                                               \
        /* perform a floorf */                                                 \
        auto tmp = __riscv_vfcvt_f_x_v_f##TLEN##m##LMUL(                       \
            __riscv_vfcvt_x_f_v_i##TLEN##m##LMUL(a1, vl), vl);                 \
        auto mask = __riscv_vmfgt_vv_f##TLEN##m##LMUL##_b##MLEN(tmp, a1, vl);  \
        tmp = __riscv_vfsub_vf_f##TLEN##m##LMUL##_m(mask, tmp, 1.f, vl);       \
                                                                               \
        auto b1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p0, vl);      \
        x = __riscv_vfnmsac_vf_f##TLEN##m##LMUL(x, c_cephes_exp_C1, tmp, vl);  \
        auto a2 = __riscv_vfcvt_x_f_v_i##TLEN##m##LMUL(tmp, vl);               \
        auto b2 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p2, vl);      \
        x = __riscv_vfnmsac_vf_f##TLEN##m##LMUL(x, c_cephes_exp_C2, tmp, vl);  \
        auto b3 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p4, vl);      \
        auto x2 = __riscv_vfmul_vv_f##TLEN##m##LMUL(x, x, vl);                 \
        b1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b1, x, c1, vl);                \
        b2 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b2, x, c3, vl);                \
        b3 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b3, x, c5, vl);                \
        b1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b1, x2, b2, vl);               \
        x = __riscv_vfadd_vf_f##TLEN##m##LMUL(x, 1.f, vl);                     \
        b1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b1, x2, b3, vl);               \
        auto a = __riscv_vsll_vx_i##TLEN##m##LMUL(a2, M, vl);                  \
        b1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(b1, x2, x, vl);                \
        auto b =                                                               \
            __riscv_vreinterpret_v_f##TLEN##m##LMUL##_i##TLEN##m##LMUL(b1);    \
                                                                               \
        /* build 2^n */                                                        \
        auto ret = __riscv_vadd_vv_i##TLEN##m##LMUL(a, b, vl);                 \
        return __riscv_vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(     \
            ret);                                                              \
    }

_RVV_FLOAT_EXP_OP(1, 32, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(2, 16, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(4, 8, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(8, 4, 32, 0x7f, 23)

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0 -1.9515295891E-4
#define c_sincof_p1 8.3321608736E-3
#define c_sincof_p2 -1.6666654611E-1
#define c_coscof_p0 2.443315711809948E-005
#define c_coscof_p1 -1.388731625493765E-003
#define c_coscof_p2 4.166664568298827E-002
#define c_cephes_FOPI 1.27323954473516 // 4 / M_PI

#define c_tanh_tiny 1e-4f
#define c_tanh_hi 9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1 4.89352455891786e-3f
#define c_tanh_alpha_3 6.37261928875436e-4f
#define c_tanh_alpha_5 1.48572235717979e-5f
#define c_tanh_alpha_7 5.12229709037114e-8f
#define c_tanh_alpha_9 -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

/*
y = p1 * x + p3 * x^3 + p5 * x^5 + p7 * x^7 + p9 * x^9 + p11 * x^11 + p13 * x^13
  = x * (p1 + p3 * x^2 + x^4 * (p5 + p7 * x^2 + x^4 * (p9 + p11 * x^2 + p13 *
x^4)))

w = p0 + p2 * x^2 + p4 * x^4 + p6 * x^6
  = p0 + p2 * x^2 + x^4 * (p4 + p6 * x^2)
*/
#define _RVV_FLOAT_TANH_OP(LMUL, MLEN, TLEN)                                   \
    static inline vfloat##TLEN##m##LMUL##_t tanh_ps(                           \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        auto abs = __riscv_vfabs_v_f##TLEN##m##LMUL(x, vl);                    \
                                                                               \
        /* clamp the inputs to the range [-9, 9] since anything outside */     \
        /* this range is -/+1.0f in single-precision.                   */     \
        abs = __riscv_vfmin_vf_f##TLEN##m##LMUL(abs, c_tanh_hi, vl);           \
                                                                               \
        /* since the polynomials are odd/even, we need x**2. */                \
        auto x2 = __riscv_vfmul_vv_f##TLEN##m##LMUL(abs, abs, vl);             \
                                                                               \
        /* evaluate the numerator polynomial y, denominator polynomial w. */   \
        auto c0 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_tanh_beta_0, vl);        \
        auto c1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_1, vl);       \
        auto c4 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_tanh_beta_4, vl);        \
        auto c5 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_5, vl);       \
        auto c9 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_9, vl);       \
        auto y1 = __riscv_vmv_v_v_f##TLEN##m##LMUL(x2, vl);                    \
        auto y2 = __riscv_vmv_v_v_f##TLEN##m##LMUL(x2, vl);                    \
        auto y3 = __riscv_vmv_v_v_f##TLEN##m##LMUL(x2, vl);                    \
        auto w1 = __riscv_vmv_v_v_f##TLEN##m##LMUL(x2, vl);                    \
        auto w2 = __riscv_vmv_v_v_f##TLEN##m##LMUL(x2, vl);                    \
        y1 = __riscv_vfmadd_vf_f##TLEN##m##LMUL(y1, c_tanh_alpha_11, c9, vl);  \
        w1 = __riscv_vfmadd_vf_f##TLEN##m##LMUL(w1, c_tanh_beta_6, c4, vl);    \
        auto x4 = __riscv_vfmul_vv_f##TLEN##m##LMUL(x2, x2, vl);               \
        y1 = __riscv_vfmacc_vf_f##TLEN##m##LMUL(y1, c_tanh_alpha_13, x4, vl);  \
        y2 = __riscv_vfmadd_vf_f##TLEN##m##LMUL(y2, c_tanh_alpha_7, c5, vl);   \
        y1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(y1, x4, y2, vl);               \
        w2 = __riscv_vfmadd_vf_f##TLEN##m##LMUL(w2, c_tanh_beta_2, c0, vl);    \
        y3 = __riscv_vfmadd_vf_f##TLEN##m##LMUL(y3, c_tanh_alpha_3, c1, vl);   \
        auto w = __riscv_vfmadd_vv_f##TLEN##m##LMUL(w1, x4, w2, vl);           \
        y1 = __riscv_vfmadd_vv_f##TLEN##m##LMUL(y1, x4, y3, vl);               \
        auto z = __riscv_vfsgnj_vv_f##TLEN##m##LMUL(abs, x, vl);               \
        w = __riscv_vfrec7_v_f##TLEN##m##LMUL(w, vl);                          \
        y1 = __riscv_vfmul_vv_f##TLEN##m##LMUL(y1, z, vl);                     \
        auto tiny_mask =                                                       \
            __riscv_vmfge_vf_f##TLEN##m##LMUL##_b##MLEN(abs, c_tanh_tiny, vl); \
                                                                               \
        /* divide the numerator by the denominator. */                         \
        auto y = __riscv_vfmul_vv_f##TLEN##m##LMUL(y1, w, vl);                 \
                                                                               \
        /* when the argument is very small in magnitude it's more accurate to  \
         * just return it. */                                                  \
        y = __riscv_vmerge_vvm_f##TLEN##m##LMUL(x, y, tiny_mask, vl);          \
                                                                               \
        return y;                                                              \
    }

_RVV_FLOAT_TANH_OP(1, 32, 32)
_RVV_FLOAT_TANH_OP(2, 16, 32)
_RVV_FLOAT_TANH_OP(4, 8, 32)
_RVV_FLOAT_TANH_OP(8, 4, 32)

#define _RVV_FLOAT_POW_OP(LMUL, MLEN, TLEN)                                    \
    static inline vfloat##TLEN##m##LMUL##_t pow_ps(                            \
        vfloat##TLEN##m##LMUL##_t a, vfloat##TLEN##m##LMUL##_t b, size_t vl) { \
        /* pow(x, m) = exp(m * log(x)) */                                      \
        return exp_ps(__riscv_vfmul_vv_f##TLEN##m##LMUL(b, log_ps(a, vl), vl), \
                      vl);                                                     \
    }

_RVV_FLOAT_POW_OP(1, 32, 32)
_RVV_FLOAT_POW_OP(2, 16, 32)
_RVV_FLOAT_POW_OP(4, 8, 32)
_RVV_FLOAT_POW_OP(8, 4, 32)

#endif