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

#if 0
// from glibc 2.40: max_ulp_error = 3
// e^x -1 = x + 1/2!x^2 + 1/3!x^3 + 1/4!x^4 + 1/5!x^5 + 1/6!x^6 + 1/7!x^7
#define _RVV_FLOAT_EXPM1F_OP(LMUL, MLEN, TLEN, E, M)                           \
    static inline vfloat##TLEN##m##LMUL##_t expm1f(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        /* Reduce argument to smaller range:                                   \
            Let i = round(x / ln2)                                             \
            and f = x - i * ln2, then f is in [-ln2/2, ln2/2].                 \
            exp(x) - 1 = 2^i * (expm1(f) + 1) - 1                              \
            where 2^i is exact because i is an integer.  */                    \
        auto shift = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.8p23f, vl);         \
        auto j = __riscv_vmv_v_v_f32m##LMUL(x, vl);                            \
        j = __riscv_vfmadd_vf_f##TLEN##m##LMUL(j, 0x1.715476p+0f, shift, vl);  \
        j = __riscv_vfsub_vv_f##TLEN##m##LMUL(j, shift, vl);                   \
        auto f = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.62e4p-1f, vl);          \
        auto c0 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.7f7d1cp-20f, vl);      \
        f = __riscv_vfnmsub_vv_f##TLEN##m##LMUL(f, j, x, vl);                  \
        auto i = __riscv_vfcvt_x_f_v_i##TLEN##m##LMUL(j, vl);                  \
        f = __riscv_vfnmsac_vv_f##TLEN##m##LMUL(f, j, c0, vl);                 \
        /* Approximate expm1(f) using polynomial.                              \
            Taylor expansion for expm1(x) has the form:                        \
            x + ax^2 + bx^3 + cx^4 ....                                        \
            So we calculate the polynomial P(f) = a + bf + cf^2 + ...          \
            and assemble the approximation expm1(f) ~= f + f^2 * P(f).  */     \
        auto poly_0 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.fffffep-2, vl);    \
        auto poly_1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.5554aep-3, vl);    \
        auto poly_2 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.555736p-5, vl);    \
        auto poly_3 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.12287cp-7, vl);    \
        auto poly_4 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.6b55a2p-10, vl);   \
        auto p = __riscv_vfmadd_vv_f##TLEN##m##LMUL(poly_4, f, poly_3, vl);    \
        auto f2 = __riscv_vfmul_vv_f##TLEN##m##LMUL(f, f, vl);                 \
        p = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p, f, poly_2, vl);              \
        p = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p, f, poly_1, vl);              \
        p = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p, f, poly_0, vl);              \
        p = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p, f2, f, vl);                  \
        auto u = __riscv_vsll_vx_i##TLEN##m##LMUL(i, 23, vl);                  \
        u = __riscv_vadd_vx_i32m##LMUL(u, 0x3f800000, vl);                     \
        /* expm1(x) ~= p * t + (t - 1).  */                                    \
        auto t =                                                               \
            __riscv_vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(u);     \
        auto tmp = __riscv_vfsub_vf_f##TLEN##m##LMUL(t, 1.f, vl);              \
        return __riscv_vfmadd_vv_f##TLEN##m##LMUL(p, t, tmp, vl);              \
    }

_RVV_FLOAT_EXPM1F_OP(1, 32, 32, 0x7f, 23)
_RVV_FLOAT_EXPM1F_OP(2, 16, 32, 0x7f, 23)
_RVV_FLOAT_EXPM1F_OP(4, 8, 32, 0x7f, 23)
_RVV_FLOAT_EXPM1F_OP(8, 4, 32, 0x7f, 23)

#define c_tanh_tiny 1e-4f
#define c_tanh_hi 9.0f
#define _RVV_FLOAT_TANH_OP(LMUL, MLEN, TLEN)                                   \
    static inline vfloat##TLEN##m##LMUL##_t tanh_ps(                           \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        auto abs = __riscv_vfabs_v_f##TLEN##m##LMUL(x, vl);                    \
                                                                               \
        /* clamp the inputs to the range [-9, 9] since anything outside */     \
        /* this range is -/+1.0f in single-precision.                   */     \
        abs = __riscv_vfmin_vf_f##TLEN##m##LMUL(abs, c_tanh_hi, vl);           \
                                                                               \
        auto q = expm1f(__riscv_vfmul_vf_f##TLEN##m##LMUL(x, 2.f, vl), vl);    \
        auto y = __riscv_vfdiv_vv_f##TLEN##m##LMUL(                            \
            q, __riscv_vfadd_vf_f##TLEN##m##LMUL(q, 2.f, vl), vl);             \
                                                                               \
        auto tiny_mask =                                                       \
            __riscv_vmfge_vf_f##TLEN##m##LMUL##_b##MLEN(abs, c_tanh_tiny, vl); \
                                                                               \
        /* when the argument is very small in magnitude it's more accurate to  \
         * just return it. */                                                  \
        y = __riscv_vmerge_vvm_f##TLEN##m##LMUL(x, y, tiny_mask, vl);          \
                                                                               \
        return y;                                                              \
    }

#else
#define LOG2_INV 0x1.71547652b82fep+0
#define LOG2_HI 0x1.62e42fefa39efp-1
#define LOG2_LO 0x1.abc9e3b39803fp-56
#define _RVV_FLOAT_TANH_OP(LMUL, MLEN, TLEN)                                   \
    static inline vfloat##TLEN##m##LMUL##_t tanh_ps(                           \
        vfloat##TLEN##m##LMUL##_t v, size_t vl) {                              \
        constexpr float fp_posZero = 0.0f;                                     \
        constexpr float fp_posOne = 1.f;                                       \
        auto zero = __riscv_vfmv_v_f_f##TLEN##m##LMUL(fp_posZero, vl);         \
        auto one = __riscv_vfmv_v_f_f##TLEN##m##LMUL(fp_posOne, vl);           \
        /*tanh(x) = sign(x) * tanh(|x|); suffices to work on |x| for the main  \
         * part */                                                             \
        auto vx = __riscv_vfsgnj_vf_f##TLEN####m##LMUL(v, 1.f, vl);            \
        /* Suffices to clip |x| to 20, which is bigger than 28 log(2) */       \
        vx = __riscv_vfmin_vf_f##TLEN##m##LMUL(vx, 0x1.4p4, vl);               \
                                                                               \
        /* tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)); so we compute exp(-2x)   \
         */                                                                    \
        /* by replacing x by -2x */                                            \
        vx = __riscv_vfmul_vf_f##TLEN##m##LMUL(vx, -2.f, vl);                  \
        auto n_flt = __riscv_vfmul_vf_f##TLEN##m##LMUL(vx, LOG2_INV, vl);      \
        auto n = __riscv_vfcvt_x_f_v_i##TLEN##m##LMUL(n_flt, vl);              \
        n_flt = __riscv_vfcvt_f_x_v_f##TLEN##m##LMUL(n, vl);                   \
        auto u = __riscv_vadd_vx_i##TLEN##m##LMUL(n, 127, vl);                 \
        auto r_delta =                                                         \
            __riscv_vfnmsac_vf_f##TLEN##m##LMUL(vx, LOG2_HI, n_flt, vl);       \
        u = __riscv_vsll_vx_i##TLEN####m##LMUL(u, 23, vl);                     \
        auto r =                                                               \
            __riscv_vfnmsac_vf_f##TLEN##m##LMUL(r_delta, LOG2_LO, n_flt, vl);  \
        auto s =                                                               \
            __riscv_vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(u);     \
        auto s_is_small =                                                      \
            __riscv_vmsle_vx_i##TLEN##m##LMUL##_b##MLEN(n, -(23 + 1), vl);     \
        r_delta = __riscv_vfsub_vv_f##TLEN##m##LMUL(r_delta, r, vl);           \
        auto s_head = __riscv_vfmerge_vfm_f##TLEN##m##LMUL(s, fp_posZero,      \
                                                           s_is_small, vl);    \
        r_delta =                                                              \
            __riscv_vfnmsac_vf_f##TLEN##m##LMUL(r_delta, LOG2_LO, n_flt, vl);  \
        /* exp(x) = 2^n exp(r'), r' = r + r_delta and thus we compute 1 +/-    \
        exp(x) as 1 +/- 2^(n)(1 + r' + (r')^2/2 + r^3 p(r)) (1 +/- s) +/- s(r' \
        + (r')^2/2) +/- s r^3 p(r) To maintain good precision, 1 +/- s and r'  \
        + (r')^2/2 are computed to extra precision in a leading term and a     \
        correctional term. This leads to representing 1 +/- exp(x) in a        \
        leading and correctional term. */                                      \
        /* 1 +/- s is exact when s is not small */                             \
        auto rsq = __riscv_vfmul_vv_f##TLEN##m##LMUL(r, r, vl);                \
        auto s_tail =                                                          \
            __riscv_vmerge_vvm_f##TLEN##m##LMUL(zero, s, s_is_small, vl);      \
        /* s_head + s_tail = s; and 1 +/- s is (1 +/- s_head) +/- s_tail */    \
        /* exp(r') is approximated by 1 + r' + (r')^2/2 + r^3(p_even(r^2) +    \
           r*p_odd(r^2)) using r without delta_r sufficies from the third      \
         order onwards */                                                      \
        auto rcube = __riscv_vfmul_vv_f##TLEN##m##LMUL(rsq, r, vl);            \
        auto c0 =                                                              \
            __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.71ddef82f4beep-19, vl);      \
        auto c1 =                                                              \
            __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.a01a01b32b633p-13, vl);      \
        auto c2 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.111111110ef6ap-7, vl); \
        auto c3 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.555555555555ap-3, vl); \
        auto c4 =                                                              \
            __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.a019b37a2b3dfp-16, vl);      \
        auto c5 =                                                              \
            __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.6c16c17a09506p-10, vl);      \
        auto c6 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(0x1.5555555553aefp-5, vl); \
                                                                               \
        auto p_even = __riscv_vmv_v_v_f##TLEN##m##LMUL(rsq, vl);               \
        p_even = __riscv_vfmadd_vf_f##TLEN##m##LMUL(                           \
            p_even, 0x1.af6eacd796f0bp-26, c0, vl);                            \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c1, vl);      \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c2, vl);      \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c3, vl);      \
                                                                               \
        auto p_odd = __riscv_vmv_v_v_f##TLEN##m##LMUL(rsq, vl);                \
        p_odd = __riscv_vfmadd_vf_f##TLEN##m##LMUL(                            \
            p_odd, 0x1.289788d8bdadfp-22, c4, vl);                             \
        p_odd = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, rsq, c5, vl);        \
        p_odd = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, rsq, c6, vl);        \
        auto poly = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, r, p_even, vl);  \
                                                                               \
        /* r^3 * poly will be r^3(...)                                         \
           we delay this multiplication with r^3 for now */                    \
                                                                               \
        /*  Compute r' + (r')^2/2 extra precisely */                           \
        auto r_prime = __riscv_vfmul_vf_f##TLEN##m##LMUL(r, 0x1.0p-1, vl);     \
        auto B = __riscv_vfmadd_vv_f##TLEN##m##LMUL(r, r_prime, r, vl);        \
        auto b = __riscv_vfsub_vv_f##TLEN##m##LMUL(r, B, vl);                  \
        b = __riscv_vfmacc_vv_f##TLEN##m##LMUL(b, r, r_prime, vl);             \
        /* B + b is r' + (r')^2/2 extra precisely */                           \
        /* incoporate r_delta in R + R^2/2 */                                  \
        auto c = __riscv_vfmadd_vv_f##TLEN##m##LMUL(r, r_delta, r_delta, vl);  \
        b = __riscv_vfadd_vv_f##TLEN##m##LMUL(b, c, vl);                       \
        poly = __riscv_vfmadd_vv_f##TLEN##m##LMUL(poly, rcube, b, vl);         \
        /* B + poly is r' + (r')^2/2 + r^3(.....) */                           \
        /* and exp(r') is well approximated by s*(1 + B + poly) */             \
                                                                               \
        /* We compute the denominator 1 + exp(R) first as                      \
           we will need to recipricate afterwards, the latency of which        \
           can be hidden somewhat by proceeding with the numerator             \
           at that time */                                                     \
        auto Z = __riscv_vfadd_vf_f##TLEN##m##LMUL(s_head, fp_posOne, vl);     \
        auto D_tmp = __riscv_vfmadd_vv_f##TLEN##m##LMUL(B, s, Z, vl);          \
        auto d_tmp = __riscv_vfsub_vv_f##TLEN##m##LMUL(Z, D_tmp, vl);          \
        d_tmp = __riscv_vfmacc_vv_f##TLEN##m##LMUL(d_tmp, s, B, vl);           \
        d_tmp = __riscv_vfadd_vv_f##TLEN##m##LMUL(d_tmp, s_tail, vl);          \
        d_tmp = __riscv_vfmacc_vv_f##TLEN##m##LMUL(d_tmp, s, poly, vl);        \
        /* D_tmp + d_tmp is 1 + exp(R) to high precision, but we have to       \
           normalize this representation so that the leading term              \
           has full FP64 precision of this sum */                              \
        auto D = __riscv_vfadd_vv_f##TLEN##m##LMUL(D_tmp, d_tmp, vl);          \
        auto d = __riscv_vfsub_vv_f##TLEN##m##LMUL(D_tmp, D, vl);              \
        d = __riscv_vfadd_vv_f##TLEN##m##LMUL(d, d_tmp, vl);                   \
                                                                               \
        /* Now start to compute 1/(D+d) as E + e */                            \
        auto E = __riscv_vfrdiv_vf_f##TLEN##m##LMUL(D, fp_posOne, vl);         \
        auto e = __riscv_vfnmsub_vv_f##TLEN##m##LMUL(E, D, one, vl);           \
        e = __riscv_vfnmsac_vv_f##TLEN##m##LMUL(e, E, d, vl);                  \
        e = __riscv_vfmul_vv_f##TLEN##m##LMUL(                                 \
            e, __riscv_vfrec7_v_f##TLEN##m##LMUL(D, vl), vl);                  \
        /* E + e is 1/(D+d) to extra precision */                              \
                                                                               \
        /* Overlap much of the 1/(D+d) computation with */                     \
        /* computing 1 - s(1 + B + poly) */                                    \
        Z = __riscv_vfrsub_vf_f##TLEN##m##LMUL(s_head, fp_posOne, vl);         \
                                                                               \
        auto Numer = __riscv_vfnmsub_vv_f##TLEN##m##LMUL(B, s, Z, vl);         \
        auto numer = __riscv_vfsub_vv_f##TLEN##m##LMUL(Z, Numer, vl);          \
        numer = __riscv_vfnmsac_vv_f##TLEN##m##LMUL(numer, s, B, vl);          \
                                                                               \
        /* Numer + numer = Z - s * B accurately */                             \
        numer = __riscv_vfsub_vv_f##TLEN##m##LMUL(numer, s_tail, vl);          \
        numer = __riscv_vfnmsac_vv_f##TLEN##m##LMUL(numer, s, poly, vl);       \
                                                                               \
        /* (Numer + numer) * (E + e) */                                        \
        /* Numer * E + ( numer * E + (Numer * e + (e*numer)) ) */              \
        auto vy = __riscv_vfmul_vv_f##TLEN##m##LMUL(e, numer, vl);             \
        vy = __riscv_vfmacc_vv_f##TLEN##m##LMUL(vy, Numer, e, vl);             \
        vy = __riscv_vfmacc_vv_f##TLEN##m##LMUL(vy, numer, E, vl);             \
        vy = __riscv_vfmacc_vv_f##TLEN##m##LMUL(vy, Numer, E, vl);             \
        return __riscv_vfsgnj_vv_f##TLEN####m##LMUL(vy, v, vl);                \
    }
#endif

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

struct sv_erff_data {
    float erf[513];
    float scale[513];
};

/* Lookup table used in SVE erff.
   For each possible rounded input r (multiples of 1/128), between
   r = 0.0 and r = 4.0 (513 values):
   - __erff_data.erf contains the values of erf(r),
   - __erff_data.scale contains the values of 2/sqrt(pi)*exp(-r^2).
   Note that indices 0 and 1 are never hit by the algorithm, since lookup is
   performed only for x >= 1/64-1/512.  */
const struct sv_erff_data __sv_erff_data = {
    .erf =
        {
            0x0.000000p+0, 0x0.000000p+0, 0x1.20d770p-6, 0x1.b137e0p-6,
            0x1.20c564p-5, 0x1.68e5d4p-5, 0x1.b0fafep-5, 0x1.f902a8p-5,
            0x1.207d48p-4, 0x1.44703ep-4, 0x1.68591ap-4, 0x1.8c36bep-4,
            0x1.b00812p-4, 0x1.d3cbf8p-4, 0x1.f7815ap-4, 0x1.0d9390p-3,
            0x1.1f5e1ap-3, 0x1.311fc2p-3, 0x1.42d7fcp-3, 0x1.548642p-3,
            0x1.662a0cp-3, 0x1.77c2d2p-3, 0x1.895010p-3, 0x1.9ad142p-3,
            0x1.ac45e4p-3, 0x1.bdad72p-3, 0x1.cf076ep-3, 0x1.e05354p-3,
            0x1.f190aap-3, 0x1.015f78p-2, 0x1.09eed6p-2, 0x1.127632p-2,
            0x1.1af54ep-2, 0x1.236bf0p-2, 0x1.2bd9dcp-2, 0x1.343ed6p-2,
            0x1.3c9aa8p-2, 0x1.44ed18p-2, 0x1.4d35f0p-2, 0x1.5574f4p-2,
            0x1.5da9f4p-2, 0x1.65d4b8p-2, 0x1.6df50ap-2, 0x1.760abap-2,
            0x1.7e1594p-2, 0x1.861566p-2, 0x1.8e0a02p-2, 0x1.95f336p-2,
            0x1.9dd0d2p-2, 0x1.a5a2acp-2, 0x1.ad6896p-2, 0x1.b52264p-2,
            0x1.bccfecp-2, 0x1.c47104p-2, 0x1.cc0584p-2, 0x1.d38d44p-2,
            0x1.db081cp-2, 0x1.e275eap-2, 0x1.e9d68ap-2, 0x1.f129d4p-2,
            0x1.f86faap-2, 0x1.ffa7eap-2, 0x1.03693ap-1, 0x1.06f794p-1,
            0x1.0a7ef6p-1, 0x1.0dff50p-1, 0x1.117894p-1, 0x1.14eab4p-1,
            0x1.1855a6p-1, 0x1.1bb95cp-1, 0x1.1f15ccp-1, 0x1.226ae8p-1,
            0x1.25b8a8p-1, 0x1.28ff02p-1, 0x1.2c3decp-1, 0x1.2f755cp-1,
            0x1.32a54cp-1, 0x1.35cdb4p-1, 0x1.38ee8ap-1, 0x1.3c07cap-1,
            0x1.3f196ep-1, 0x1.42236ep-1, 0x1.4525c8p-1, 0x1.482074p-1,
            0x1.4b1372p-1, 0x1.4dfebap-1, 0x1.50e24cp-1, 0x1.53be26p-1,
            0x1.569244p-1, 0x1.595ea6p-1, 0x1.5c2348p-1, 0x1.5ee02ep-1,
            0x1.619556p-1, 0x1.6442c0p-1, 0x1.66e86ep-1, 0x1.69865ep-1,
            0x1.6c1c98p-1, 0x1.6eab18p-1, 0x1.7131e6p-1, 0x1.73b102p-1,
            0x1.762870p-1, 0x1.789836p-1, 0x1.7b0058p-1, 0x1.7d60d8p-1,
            0x1.7fb9c0p-1, 0x1.820b12p-1, 0x1.8454d6p-1, 0x1.869712p-1,
            0x1.88d1cep-1, 0x1.8b050ep-1, 0x1.8d30dep-1, 0x1.8f5544p-1,
            0x1.91724ap-1, 0x1.9387f6p-1, 0x1.959652p-1, 0x1.979d68p-1,
            0x1.999d42p-1, 0x1.9b95e8p-1, 0x1.9d8768p-1, 0x1.9f71cap-1,
            0x1.a1551ap-1, 0x1.a33162p-1, 0x1.a506b0p-1, 0x1.a6d50cp-1,
            0x1.a89c86p-1, 0x1.aa5d26p-1, 0x1.ac16fcp-1, 0x1.adca14p-1,
            0x1.af767ap-1, 0x1.b11c3cp-1, 0x1.b2bb68p-1, 0x1.b4540ap-1,
            0x1.b5e630p-1, 0x1.b771e8p-1, 0x1.b8f742p-1, 0x1.ba764ap-1,
            0x1.bbef10p-1, 0x1.bd61a2p-1, 0x1.bece0ep-1, 0x1.c03464p-1,
            0x1.c194b2p-1, 0x1.c2ef08p-1, 0x1.c44376p-1, 0x1.c5920ap-1,
            0x1.c6dad2p-1, 0x1.c81de2p-1, 0x1.c95b46p-1, 0x1.ca930ep-1,
            0x1.cbc54cp-1, 0x1.ccf20cp-1, 0x1.ce1962p-1, 0x1.cf3b5cp-1,
            0x1.d0580cp-1, 0x1.d16f7ep-1, 0x1.d281c4p-1, 0x1.d38ef0p-1,
            0x1.d49710p-1, 0x1.d59a34p-1, 0x1.d6986cp-1, 0x1.d791cap-1,
            0x1.d8865ep-1, 0x1.d97636p-1, 0x1.da6162p-1, 0x1.db47f4p-1,
            0x1.dc29fcp-1, 0x1.dd0788p-1, 0x1.dde0aap-1, 0x1.deb570p-1,
            0x1.df85eap-1, 0x1.e0522ap-1, 0x1.e11a3ep-1, 0x1.e1de36p-1,
            0x1.e29e22p-1, 0x1.e35a12p-1, 0x1.e41214p-1, 0x1.e4c638p-1,
            0x1.e5768cp-1, 0x1.e62322p-1, 0x1.e6cc08p-1, 0x1.e7714ap-1,
            0x1.e812fcp-1, 0x1.e8b12ap-1, 0x1.e94be4p-1, 0x1.e9e336p-1,
            0x1.ea7730p-1, 0x1.eb07e2p-1, 0x1.eb9558p-1, 0x1.ec1fa2p-1,
            0x1.eca6ccp-1, 0x1.ed2ae6p-1, 0x1.edabfcp-1, 0x1.ee2a1ep-1,
            0x1.eea556p-1, 0x1.ef1db4p-1, 0x1.ef9344p-1, 0x1.f00614p-1,
            0x1.f07630p-1, 0x1.f0e3a6p-1, 0x1.f14e82p-1, 0x1.f1b6d0p-1,
            0x1.f21ca0p-1, 0x1.f27ff8p-1, 0x1.f2e0eap-1, 0x1.f33f7ep-1,
            0x1.f39bc2p-1, 0x1.f3f5c2p-1, 0x1.f44d88p-1, 0x1.f4a31ep-1,
            0x1.f4f694p-1, 0x1.f547f2p-1, 0x1.f59742p-1, 0x1.f5e490p-1,
            0x1.f62fe8p-1, 0x1.f67952p-1, 0x1.f6c0dcp-1, 0x1.f7068cp-1,
            0x1.f74a6ep-1, 0x1.f78c8cp-1, 0x1.f7cceep-1, 0x1.f80ba2p-1,
            0x1.f848acp-1, 0x1.f8841ap-1, 0x1.f8bdf2p-1, 0x1.f8f63ep-1,
            0x1.f92d08p-1, 0x1.f96256p-1, 0x1.f99634p-1, 0x1.f9c8a8p-1,
            0x1.f9f9bap-1, 0x1.fa2974p-1, 0x1.fa57dep-1, 0x1.fa84fep-1,
            0x1.fab0dep-1, 0x1.fadb84p-1, 0x1.fb04f6p-1, 0x1.fb2d40p-1,
            0x1.fb5464p-1, 0x1.fb7a6cp-1, 0x1.fb9f60p-1, 0x1.fbc344p-1,
            0x1.fbe61ep-1, 0x1.fc07fap-1, 0x1.fc28d8p-1, 0x1.fc48c2p-1,
            0x1.fc67bcp-1, 0x1.fc85d0p-1, 0x1.fca2fep-1, 0x1.fcbf52p-1,
            0x1.fcdaccp-1, 0x1.fcf576p-1, 0x1.fd0f54p-1, 0x1.fd286ap-1,
            0x1.fd40bep-1, 0x1.fd5856p-1, 0x1.fd6f34p-1, 0x1.fd8562p-1,
            0x1.fd9ae2p-1, 0x1.fdafb8p-1, 0x1.fdc3e8p-1, 0x1.fdd77ap-1,
            0x1.fdea6ep-1, 0x1.fdfcccp-1, 0x1.fe0e96p-1, 0x1.fe1fd0p-1,
            0x1.fe3080p-1, 0x1.fe40a6p-1, 0x1.fe504cp-1, 0x1.fe5f70p-1,
            0x1.fe6e18p-1, 0x1.fe7c46p-1, 0x1.fe8a00p-1, 0x1.fe9748p-1,
            0x1.fea422p-1, 0x1.feb090p-1, 0x1.febc96p-1, 0x1.fec836p-1,
            0x1.fed374p-1, 0x1.fede52p-1, 0x1.fee8d4p-1, 0x1.fef2fep-1,
            0x1.fefccep-1, 0x1.ff064cp-1, 0x1.ff0f76p-1, 0x1.ff1852p-1,
            0x1.ff20e0p-1, 0x1.ff2924p-1, 0x1.ff3120p-1, 0x1.ff38d6p-1,
            0x1.ff4048p-1, 0x1.ff4778p-1, 0x1.ff4e68p-1, 0x1.ff551ap-1,
            0x1.ff5b90p-1, 0x1.ff61ccp-1, 0x1.ff67d0p-1, 0x1.ff6d9ep-1,
            0x1.ff7338p-1, 0x1.ff789ep-1, 0x1.ff7dd4p-1, 0x1.ff82dap-1,
            0x1.ff87b2p-1, 0x1.ff8c5cp-1, 0x1.ff90dcp-1, 0x1.ff9532p-1,
            0x1.ff9960p-1, 0x1.ff9d68p-1, 0x1.ffa14ap-1, 0x1.ffa506p-1,
            0x1.ffa8a0p-1, 0x1.ffac18p-1, 0x1.ffaf6ep-1, 0x1.ffb2a6p-1,
            0x1.ffb5bep-1, 0x1.ffb8b8p-1, 0x1.ffbb98p-1, 0x1.ffbe5ap-1,
            0x1.ffc102p-1, 0x1.ffc390p-1, 0x1.ffc606p-1, 0x1.ffc862p-1,
            0x1.ffcaa8p-1, 0x1.ffccd8p-1, 0x1.ffcef4p-1, 0x1.ffd0fap-1,
            0x1.ffd2eap-1, 0x1.ffd4cap-1, 0x1.ffd696p-1, 0x1.ffd84ep-1,
            0x1.ffd9f8p-1, 0x1.ffdb90p-1, 0x1.ffdd18p-1, 0x1.ffde90p-1,
            0x1.ffdffap-1, 0x1.ffe154p-1, 0x1.ffe2a2p-1, 0x1.ffe3e2p-1,
            0x1.ffe514p-1, 0x1.ffe63cp-1, 0x1.ffe756p-1, 0x1.ffe866p-1,
            0x1.ffe96ap-1, 0x1.ffea64p-1, 0x1.ffeb54p-1, 0x1.ffec3ap-1,
            0x1.ffed16p-1, 0x1.ffedeap-1, 0x1.ffeeb4p-1, 0x1.ffef76p-1,
            0x1.fff032p-1, 0x1.fff0e4p-1, 0x1.fff18ep-1, 0x1.fff232p-1,
            0x1.fff2d0p-1, 0x1.fff366p-1, 0x1.fff3f6p-1, 0x1.fff480p-1,
            0x1.fff504p-1, 0x1.fff582p-1, 0x1.fff5fcp-1, 0x1.fff670p-1,
            0x1.fff6dep-1, 0x1.fff74ap-1, 0x1.fff7aep-1, 0x1.fff810p-1,
            0x1.fff86cp-1, 0x1.fff8c6p-1, 0x1.fff91cp-1, 0x1.fff96cp-1,
            0x1.fff9bap-1, 0x1.fffa04p-1, 0x1.fffa4cp-1, 0x1.fffa90p-1,
            0x1.fffad0p-1, 0x1.fffb0ep-1, 0x1.fffb4ap-1, 0x1.fffb82p-1,
            0x1.fffbb8p-1, 0x1.fffbecp-1, 0x1.fffc1ep-1, 0x1.fffc4ep-1,
            0x1.fffc7ap-1, 0x1.fffca6p-1, 0x1.fffccep-1, 0x1.fffcf6p-1,
            0x1.fffd1ap-1, 0x1.fffd3ep-1, 0x1.fffd60p-1, 0x1.fffd80p-1,
            0x1.fffda0p-1, 0x1.fffdbep-1, 0x1.fffddap-1, 0x1.fffdf4p-1,
            0x1.fffe0ep-1, 0x1.fffe26p-1, 0x1.fffe3ep-1, 0x1.fffe54p-1,
            0x1.fffe68p-1, 0x1.fffe7ep-1, 0x1.fffe90p-1, 0x1.fffea2p-1,
            0x1.fffeb4p-1, 0x1.fffec4p-1, 0x1.fffed4p-1, 0x1.fffee4p-1,
            0x1.fffef2p-1, 0x1.ffff00p-1, 0x1.ffff0cp-1, 0x1.ffff18p-1,
            0x1.ffff24p-1, 0x1.ffff30p-1, 0x1.ffff3ap-1, 0x1.ffff44p-1,
            0x1.ffff4ep-1, 0x1.ffff56p-1, 0x1.ffff60p-1, 0x1.ffff68p-1,
            0x1.ffff70p-1, 0x1.ffff78p-1, 0x1.ffff7ep-1, 0x1.ffff84p-1,
            0x1.ffff8cp-1, 0x1.ffff92p-1, 0x1.ffff98p-1, 0x1.ffff9cp-1,
            0x1.ffffa2p-1, 0x1.ffffa6p-1, 0x1.ffffacp-1, 0x1.ffffb0p-1,
            0x1.ffffb4p-1, 0x1.ffffb8p-1, 0x1.ffffbcp-1, 0x1.ffffc0p-1,
            0x1.ffffc4p-1, 0x1.ffffc6p-1, 0x1.ffffcap-1, 0x1.ffffccp-1,
            0x1.ffffd0p-1, 0x1.ffffd2p-1, 0x1.ffffd4p-1, 0x1.ffffd6p-1,
            0x1.ffffd8p-1, 0x1.ffffdcp-1, 0x1.ffffdep-1, 0x1.ffffdep-1,
            0x1.ffffe0p-1, 0x1.ffffe2p-1, 0x1.ffffe4p-1, 0x1.ffffe6p-1,
            0x1.ffffe8p-1, 0x1.ffffe8p-1, 0x1.ffffeap-1, 0x1.ffffeap-1,
            0x1.ffffecp-1, 0x1.ffffeep-1, 0x1.ffffeep-1, 0x1.fffff0p-1,
            0x1.fffff0p-1, 0x1.fffff2p-1, 0x1.fffff2p-1, 0x1.fffff2p-1,
            0x1.fffff4p-1, 0x1.fffff4p-1, 0x1.fffff4p-1, 0x1.fffff6p-1,
            0x1.fffff6p-1, 0x1.fffff6p-1, 0x1.fffff8p-1, 0x1.fffff8p-1,
            0x1.fffff8p-1, 0x1.fffff8p-1, 0x1.fffffap-1, 0x1.fffffap-1,
            0x1.fffffap-1, 0x1.fffffap-1, 0x1.fffffap-1, 0x1.fffffap-1,
            0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1,
            0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0,
        },
    .scale =
        {
            0x1.20dd76p+0,  0x1.20dd76p+0,  0x1.20cb68p+0,  0x1.20b4d8p+0,
            0x1.209546p+0,  0x1.206cb4p+0,  0x1.203b26p+0,  0x1.2000a0p+0,
            0x1.1fbd28p+0,  0x1.1f70c4p+0,  0x1.1f1b7ap+0,  0x1.1ebd56p+0,
            0x1.1e565cp+0,  0x1.1de698p+0,  0x1.1d6e14p+0,  0x1.1cecdcp+0,
            0x1.1c62fap+0,  0x1.1bd07cp+0,  0x1.1b3572p+0,  0x1.1a91e6p+0,
            0x1.19e5eap+0,  0x1.19318cp+0,  0x1.1874dep+0,  0x1.17aff0p+0,
            0x1.16e2d8p+0,  0x1.160da4p+0,  0x1.153068p+0,  0x1.144b3cp+0,
            0x1.135e30p+0,  0x1.12695ep+0,  0x1.116cd8p+0,  0x1.1068bap+0,
            0x1.0f5d16p+0,  0x1.0e4a08p+0,  0x1.0d2fa6p+0,  0x1.0c0e0ap+0,
            0x1.0ae550p+0,  0x1.09b590p+0,  0x1.087ee4p+0,  0x1.07416cp+0,
            0x1.05fd3ep+0,  0x1.04b27cp+0,  0x1.036140p+0,  0x1.0209a6p+0,
            0x1.00abd0p+0,  0x1.fe8fb0p-1,  0x1.fbbbbep-1,  0x1.f8dc0ap-1,
            0x1.f5f0cep-1,  0x1.f2fa4cp-1,  0x1.eff8c4p-1,  0x1.ecec78p-1,
            0x1.e9d5a8p-1,  0x1.e6b498p-1,  0x1.e38988p-1,  0x1.e054bep-1,
            0x1.dd167cp-1,  0x1.d9cf06p-1,  0x1.d67ea2p-1,  0x1.d32592p-1,
            0x1.cfc41ep-1,  0x1.cc5a8ap-1,  0x1.c8e91cp-1,  0x1.c5701ap-1,
            0x1.c1efcap-1,  0x1.be6872p-1,  0x1.bada5ap-1,  0x1.b745c6p-1,
            0x1.b3aafcp-1,  0x1.b00a46p-1,  0x1.ac63e8p-1,  0x1.a8b828p-1,
            0x1.a5074ep-1,  0x1.a1519ep-1,  0x1.9d9762p-1,  0x1.99d8dap-1,
            0x1.961650p-1,  0x1.925008p-1,  0x1.8e8646p-1,  0x1.8ab950p-1,
            0x1.86e96ap-1,  0x1.8316d6p-1,  0x1.7f41dcp-1,  0x1.7b6abcp-1,
            0x1.7791b8p-1,  0x1.73b714p-1,  0x1.6fdb12p-1,  0x1.6bfdf0p-1,
            0x1.681ff2p-1,  0x1.644156p-1,  0x1.60625cp-1,  0x1.5c8342p-1,
            0x1.58a446p-1,  0x1.54c5a6p-1,  0x1.50e79ep-1,  0x1.4d0a68p-1,
            0x1.492e42p-1,  0x1.455366p-1,  0x1.417a0cp-1,  0x1.3da26ep-1,
            0x1.39ccc2p-1,  0x1.35f940p-1,  0x1.32281ep-1,  0x1.2e5992p-1,
            0x1.2a8dcep-1,  0x1.26c508p-1,  0x1.22ff72p-1,  0x1.1f3d3cp-1,
            0x1.1b7e98p-1,  0x1.17c3b6p-1,  0x1.140cc4p-1,  0x1.1059eep-1,
            0x1.0cab62p-1,  0x1.09014cp-1,  0x1.055bd6p-1,  0x1.01bb2cp-1,
            0x1.fc3ee6p-2,  0x1.f511aap-2,  0x1.edeeeep-2,  0x1.e6d700p-2,
            0x1.dfca26p-2,  0x1.d8c8aap-2,  0x1.d1d2d0p-2,  0x1.cae8dap-2,
            0x1.c40b08p-2,  0x1.bd3998p-2,  0x1.b674c8p-2,  0x1.afbcd4p-2,
            0x1.a911f0p-2,  0x1.a27456p-2,  0x1.9be438p-2,  0x1.9561c8p-2,
            0x1.8eed36p-2,  0x1.8886b2p-2,  0x1.822e66p-2,  0x1.7be47ap-2,
            0x1.75a91ap-2,  0x1.6f7c6ap-2,  0x1.695e8cp-2,  0x1.634fa6p-2,
            0x1.5d4fd4p-2,  0x1.575f34p-2,  0x1.517de6p-2,  0x1.4bac00p-2,
            0x1.45e99cp-2,  0x1.4036d0p-2,  0x1.3a93b2p-2,  0x1.350052p-2,
            0x1.2f7cc4p-2,  0x1.2a0916p-2,  0x1.24a554p-2,  0x1.1f518ap-2,
            0x1.1a0dc6p-2,  0x1.14da0ap-2,  0x1.0fb662p-2,  0x1.0aa2d0p-2,
            0x1.059f5ap-2,  0x1.00ac00p-2,  0x1.f79184p-3,  0x1.edeb40p-3,
            0x1.e46530p-3,  0x1.daff4ap-3,  0x1.d1b982p-3,  0x1.c893cep-3,
            0x1.bf8e1cp-3,  0x1.b6a856p-3,  0x1.ade26cp-3,  0x1.a53c42p-3,
            0x1.9cb5bep-3,  0x1.944ec2p-3,  0x1.8c0732p-3,  0x1.83deeap-3,
            0x1.7bd5c8p-3,  0x1.73eba4p-3,  0x1.6c2056p-3,  0x1.6473b6p-3,
            0x1.5ce596p-3,  0x1.5575c8p-3,  0x1.4e241ep-3,  0x1.46f066p-3,
            0x1.3fda6cp-3,  0x1.38e1fap-3,  0x1.3206dcp-3,  0x1.2b48dap-3,
            0x1.24a7b8p-3,  0x1.1e233ep-3,  0x1.17bb2cp-3,  0x1.116f48p-3,
            0x1.0b3f52p-3,  0x1.052b0cp-3,  0x1.fe6460p-4,  0x1.f2a902p-4,
            0x1.e72372p-4,  0x1.dbd32ap-4,  0x1.d0b7a0p-4,  0x1.c5d04ap-4,
            0x1.bb1c98p-4,  0x1.b09bfcp-4,  0x1.a64de6p-4,  0x1.9c31c6p-4,
            0x1.92470ap-4,  0x1.888d1ep-4,  0x1.7f036cp-4,  0x1.75a960p-4,
            0x1.6c7e64p-4,  0x1.6381e2p-4,  0x1.5ab342p-4,  0x1.5211ecp-4,
            0x1.499d48p-4,  0x1.4154bcp-4,  0x1.3937b2p-4,  0x1.31458ep-4,
            0x1.297dbap-4,  0x1.21df9ap-4,  0x1.1a6a96p-4,  0x1.131e14p-4,
            0x1.0bf97ep-4,  0x1.04fc3ap-4,  0x1.fc4b5ep-5,  0x1.eeea8cp-5,
            0x1.e1d4d0p-5,  0x1.d508fap-5,  0x1.c885e0p-5,  0x1.bc4a54p-5,
            0x1.b05530p-5,  0x1.a4a54ap-5,  0x1.99397ap-5,  0x1.8e109cp-5,
            0x1.83298ep-5,  0x1.78832cp-5,  0x1.6e1c58p-5,  0x1.63f3f6p-5,
            0x1.5a08e8p-5,  0x1.505a18p-5,  0x1.46e66cp-5,  0x1.3dacd2p-5,
            0x1.34ac36p-5,  0x1.2be38cp-5,  0x1.2351c2p-5,  0x1.1af5d2p-5,
            0x1.12ceb4p-5,  0x1.0adb60p-5,  0x1.031ad6p-5,  0x1.f7182ap-6,
            0x1.e85c44p-6,  0x1.da0006p-6,  0x1.cc0180p-6,  0x1.be5ecep-6,
            0x1.b1160ap-6,  0x1.a4255ap-6,  0x1.978ae8p-6,  0x1.8b44e6p-6,
            0x1.7f5188p-6,  0x1.73af0cp-6,  0x1.685bb6p-6,  0x1.5d55ccp-6,
            0x1.529b9ep-6,  0x1.482b84p-6,  0x1.3e03d8p-6,  0x1.3422fep-6,
            0x1.2a875cp-6,  0x1.212f62p-6,  0x1.181984p-6,  0x1.0f443ep-6,
            0x1.06ae14p-6,  0x1.fcab14p-7,  0x1.ec7262p-7,  0x1.dcaf36p-7,
            0x1.cd5ecap-7,  0x1.be7e5ap-7,  0x1.b00b38p-7,  0x1.a202bep-7,
            0x1.94624ep-7,  0x1.87275ep-7,  0x1.7a4f6ap-7,  0x1.6dd7fep-7,
            0x1.61beaep-7,  0x1.56011cp-7,  0x1.4a9cf6p-7,  0x1.3f8ff6p-7,
            0x1.34d7dcp-7,  0x1.2a727ap-7,  0x1.205dacp-7,  0x1.169756p-7,
            0x1.0d1d6ap-7,  0x1.03ede2p-7,  0x1.f60d8ap-8,  0x1.e4cc4ap-8,
            0x1.d4143ap-8,  0x1.c3e1a6p-8,  0x1.b430ecp-8,  0x1.a4fe84p-8,
            0x1.9646f4p-8,  0x1.8806d8p-8,  0x1.7a3adep-8,  0x1.6cdfccp-8,
            0x1.5ff276p-8,  0x1.536fc2p-8,  0x1.4754acp-8,  0x1.3b9e40p-8,
            0x1.30499cp-8,  0x1.2553eep-8,  0x1.1aba78p-8,  0x1.107a8cp-8,
            0x1.06918cp-8,  0x1.f9f9d0p-9,  0x1.e77448p-9,  0x1.d58da6p-9,
            0x1.c4412cp-9,  0x1.b38a3ap-9,  0x1.a36454p-9,  0x1.93cb12p-9,
            0x1.84ba30p-9,  0x1.762d84p-9,  0x1.682100p-9,  0x1.5a90b0p-9,
            0x1.4d78bcp-9,  0x1.40d564p-9,  0x1.34a306p-9,  0x1.28de12p-9,
            0x1.1d8318p-9,  0x1.128ebap-9,  0x1.07fdb4p-9,  0x1.fb99b8p-10,
            0x1.e7f232p-10, 0x1.d4fed8p-10, 0x1.c2b9d0p-10, 0x1.b11d70p-10,
            0x1.a02436p-10, 0x1.8fc8c8p-10, 0x1.8005f0p-10, 0x1.70d6a4p-10,
            0x1.6235fcp-10, 0x1.541f34p-10, 0x1.468daep-10, 0x1.397ceep-10,
            0x1.2ce898p-10, 0x1.20cc76p-10, 0x1.15246ep-10, 0x1.09ec86p-10,
            0x1.fe41cep-11, 0x1.e97ba4p-11, 0x1.d57f52p-11, 0x1.c245d4p-11,
            0x1.afc85ep-11, 0x1.9e0058p-11, 0x1.8ce75ep-11, 0x1.7c7744p-11,
            0x1.6caa0ep-11, 0x1.5d79ecp-11, 0x1.4ee142p-11, 0x1.40daa4p-11,
            0x1.3360ccp-11, 0x1.266ea8p-11, 0x1.19ff46p-11, 0x1.0e0de8p-11,
            0x1.0295f0p-11, 0x1.ef25d4p-12, 0x1.da0110p-12, 0x1.c5b542p-12,
            0x1.b23a5ap-12, 0x1.9f8894p-12, 0x1.8d986ap-12, 0x1.7c629ap-12,
            0x1.6be022p-12, 0x1.5c0a38p-12, 0x1.4cda54p-12, 0x1.3e4a24p-12,
            0x1.305390p-12, 0x1.22f0b4p-12, 0x1.161be4p-12, 0x1.09cfa4p-12,
            0x1.fc0d56p-13, 0x1.e577bcp-13, 0x1.cfd4a6p-13, 0x1.bb1a96p-13,
            0x1.a74068p-13, 0x1.943d4ap-13, 0x1.8208bcp-13, 0x1.709a8ep-13,
            0x1.5feadap-13, 0x1.4ff208p-13, 0x1.40a8c2p-13, 0x1.3207fcp-13,
            0x1.2408eap-13, 0x1.16a502p-13, 0x1.09d5f8p-13, 0x1.fb2b7ap-14,
            0x1.e3bcf4p-14, 0x1.cd5528p-14, 0x1.b7e946p-14, 0x1.a36eecp-14,
            0x1.8fdc1cp-14, 0x1.7d2738p-14, 0x1.6b4702p-14, 0x1.5a329cp-14,
            0x1.49e178p-14, 0x1.3a4b60p-14, 0x1.2b6876p-14, 0x1.1d3120p-14,
            0x1.0f9e1cp-14, 0x1.02a868p-14, 0x1.ec929ap-15, 0x1.d4f4b4p-15,
            0x1.be6abcp-15, 0x1.a8e8ccp-15, 0x1.94637ep-15, 0x1.80cfdcp-15,
            0x1.6e2368p-15, 0x1.5c540cp-15, 0x1.4b581cp-15, 0x1.3b2652p-15,
            0x1.2bb5ccp-15, 0x1.1cfe02p-15, 0x1.0ef6c4p-15, 0x1.019842p-15,
            0x1.e9b5e8p-16, 0x1.d16f58p-16, 0x1.ba4f04p-16, 0x1.a447b8p-16,
            0x1.8f4cccp-16, 0x1.7b5224p-16, 0x1.684c22p-16, 0x1.562facp-16,
            0x1.44f21ep-16, 0x1.34894ap-16, 0x1.24eb72p-16, 0x1.160f44p-16,
            0x1.07ebd2p-16, 0x1.f4f12ep-17, 0x1.db5ad0p-17, 0x1.c304f0p-17,
            0x1.abe09ep-17, 0x1.95df98p-17, 0x1.80f43ap-17, 0x1.6d1178p-17,
            0x1.5a2ae0p-17, 0x1.483488p-17, 0x1.372310p-17, 0x1.26eb9ep-17,
            0x1.1783cep-17, 0x1.08e1bap-17, 0x1.f5f7d8p-18, 0x1.db92b6p-18,
            0x1.c282cep-18, 0x1.aab7acp-18, 0x1.94219cp-18, 0x1.7eb1a2p-18,
            0x1.6a5972p-18, 0x1.570b6ap-18, 0x1.44ba86p-18, 0x1.335a62p-18,
            0x1.22df2ap-18, 0x1.133d96p-18, 0x1.046aeap-18, 0x1.ecb9d0p-19,
            0x1.d21398p-19, 0x1.b8d094p-19, 0x1.a0df10p-19, 0x1.8a2e26p-19,
            0x1.74adc8p-19, 0x1.604ea8p-19, 0x1.4d0232p-19, 0x1.3aba86p-19,
            0x1.296a70p-19, 0x1.190562p-19, 0x1.097f62p-19, 0x1.f59a20p-20,
            0x1.d9c736p-20, 0x1.bf716cp-20, 0x1.a6852cp-20, 0x1.8eefd8p-20,
            0x1.789fb8p-20, 0x1.6383f8p-20, 0x1.4f8c96p-20, 0x1.3caa62p-20,
            0x1.2acee2p-20, 0x1.19ec60p-20, 0x1.09f5d0p-20, 0x1.f5bd96p-21,
            0x1.d9371ep-21, 0x1.be41dep-21, 0x1.a4c89ep-21, 0x1.8cb738p-21,
            0x1.75fa8ep-21, 0x1.608078p-21, 0x1.4c37c0p-21, 0x1.39100ep-21,
            0x1.26f9e0p-21, 0x1.15e682p-21, 0x1.05c804p-21, 0x1.ed2254p-22,
            0x1.d06ad6p-22, 0x1.b551c8p-22, 0x1.9bc0a0p-22, 0x1.83a200p-22,
            0x1.6ce1aap-22, 0x1.576c72p-22, 0x1.43302cp-22, 0x1.301ba2p-22,
            0x1.1e1e86p-22, 0x1.0d2966p-22, 0x1.fa5b50p-23, 0x1.dc3ae4p-23,
            0x1.bfd756p-23, 0x1.a517dap-23, 0x1.8be4f8p-23, 0x1.74287ep-23,
            0x1.5dcd66p-23, 0x1.48bfd4p-23, 0x1.34ecf8p-23, 0x1.224310p-23,
            0x1.10b148p-23,
        },
};

#define _RVV_FLOAT_ERF_OP(LMUL, MLEN, TLEN)                                    \
    static inline vfloat##TLEN##m##LMUL##_t erf_ps(                            \
        fixed_vfloat##TLEN##m##LMUL##_t x, size_t vl) {                        \
        auto zero = __riscv_vmv_v_x_u##TLEN##m##LMUL(0, vl);                   \
        auto a = __riscv_vfabs_v_f##TLEN##m##LMUL(x, vl);                      \
                                                                               \
        /* |x| > 1/64 - 1/512.  */                                             \
        auto gt_min_mask =                                                     \
            __riscv_vmfgt_vf_f##TLEN##m##LMUL##_b##MLEN(a, 0x1.cp-7f, vl);     \
                                                                               \
        auto tmp_i = __riscv_vfmul_vf_f##TLEN##m##LMUL(a, 128.f, vl);          \
        auto i = __riscv_vfcvt_xu_f_v_u##TLEN##m##LMUL(tmp_i, vl);             \
                                                                               \
        /* Saturate lookup index.  */                                          \
        i = __riscv_vmerge_vvm_u##TLEN##m##LMUL(zero, i, gt_min_mask, vl);     \
        i = __riscv_vminu_vx_u##TLEN##m##LMUL(i, 512, vl);                     \
        auto tmp_r = __riscv_vfcvt_f_xu_v_f##TLEN##m##LMUL(i, vl);             \
        i = __riscv_vmul_vx_u##TLEN##m##LMUL(i, TLEN / 8, vl);                 \
                                                                               \
        /* r and erf(r) set to 0 for |x| below min.  */                        \
        auto r = __riscv_vfmul_vf_f##TLEN##m##LMUL(tmp_r, 1.f / 128, vl);      \
        auto erfr = __riscv_vluxei##TLEN##_v_f##TLEN##m##LMUL(                 \
            __sv_erff_data.erf, i, vl);                                        \
        auto scale = __riscv_vluxei##TLEN##_v_f##TLEN##m##LMUL(                \
            __sv_erff_data.scale, i, vl);                                      \
                                                                               \
        /* |x| >= 4.0 - 8/128.  */                                             \
        auto ge_max_mask =                                                     \
            __riscv_vmfge_vf_f##TLEN##m##LMUL##_b##MLEN(a, 3.9375f, vl);       \
                                                                               \
        /* erf(x) ~ erf(r) + scale * d * (1 - r * d - 1/3 * d^2).  */          \
        auto d = __riscv_vfsub_vv_f##TLEN##m##LMUL(a, r, vl);                  \
        auto d2 = __riscv_vfmul_vv_f##TLEN##m##LMUL(d, d, vl);                 \
        auto y = __riscv_vfmacc_vf_f##TLEN##m##LMUL(r, 0x1.555556p-2f, d, vl); \
        y = __riscv_vfnmsub_vv_f##TLEN##m##LMUL(y, d2, d, vl);                 \
        y = __riscv_vfmadd_vv_f##TLEN##m##LMUL(y, scale, erfr, vl);            \
                                                                               \
        /* Solves the |x| = inf case.  */                                      \
        y = __riscv_vfmerge_vfm_f##TLEN##m##LMUL(y, 1.f, ge_max_mask, vl);     \
                                                                               \
        /* Copy sign.  */                                                      \
        return __riscv_vfsgnj_vv_f##TLEN##m##LMUL(y, x, vl);                   \
    }

_RVV_FLOAT_ERF_OP(1, 32, 32)
_RVV_FLOAT_ERF_OP(2, 16, 32)
_RVV_FLOAT_ERF_OP(4, 8, 32)
_RVV_FLOAT_ERF_OP(8, 4, 32)
#endif