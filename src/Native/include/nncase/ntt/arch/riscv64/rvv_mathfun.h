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
    static inline __attribute__((                                              \
        optimize("no-schedule-insns2"))) vfloat##TLEN##m##LMUL##_t             \
    exp_ps(vfloat##TLEN##m##LMUL##_t x, size_t vl) {                           \
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
        auto zero = __riscv_vfmv_v_f_f32m##LMUL(fp_posZero, vl);               \
        auto one = __riscv_vfmv_v_f_f32m##LMUL(fp_posOne, vl);                 \
        /*tanh(x) = sign(x) * tanh(|x|); suffices to work on |x| for the main  \
         * part */                                                             \
        auto vx = __riscv_vfsgnj_vf_f32##m##LMUL(v, 1.f, vl);                  \
        /* Suffices to clip |x| to 20, which is bigger than 28 log(2) */       \
        vx = __riscv_vfmin_vf_f##TLEN##m##LMUL(vx, 0x1.4p4, vl);               \
                                                                               \
        /* tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)); so we compute exp(-2x)   \
         */                                                                    \
        /* by replacing x by -2x */                                            \
        vx = __riscv_vfmul_vf_f##TLEN##m##LMUL(vx, -2.f, vl);                  \
        auto n_flt = __riscv_vfmul_vf_f##TLEN##m##LMUL(vx, LOG2_INV, vl);      \
        auto n = __riscv_vfcvt_x_f_v_i32m##LMUL(n_flt, vl);                    \
        n_flt = __riscv_vfcvt_f_x_v_f32m##LMUL(n, vl);                         \
        auto u = __riscv_vadd_vx_i32m##LMUL(n, 127, vl);                       \
        auto r_delta = __riscv_vfnmsac_vf_f32m##LMUL(vx, LOG2_HI, n_flt, vl);  \
        u = __riscv_vsll_vx_i32##m##LMUL(u, 23, vl);                           \
        auto r = __riscv_vfnmsac_vf_f32m##LMUL(r_delta, LOG2_LO, n_flt, vl);   \
        auto s = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(u);            \
        auto s_is_small =                                                      \
            __riscv_vmsle_vx_i32m##LMUL##_b##MLEN(n, -(23 + 1), vl);           \
        r_delta = __riscv_vfsub_vv_f##TLEN##m##LMUL(r_delta, r, vl);           \
        auto s_head =                                                          \
            __riscv_vfmerge_vfm_f32m##LMUL(s, fp_posZero, s_is_small, vl);     \
        r_delta = __riscv_vfnmsac_vf_f32m##LMUL(r_delta, LOG2_LO, n_flt, vl);  \
        /* exp(x) = 2^n exp(r'), r' = r + r_delta and thus we compute 1 +/-    \
        exp(x) as 1 +/- 2^(n)(1 + r' + (r')^2/2 + r^3 p(r)) (1 +/- s) +/- s(r' \
        + (r')^2/2) +/- s r^3 p(r) To maintain good precision, 1 +/- s and r'  \
        + (r')^2/2 are computed to extra precision in a leading term and a     \
        correctional term. This leads to representing 1 +/- exp(x) in a        \
        leading and correctional term. */                                      \
        /* 1 +/- s is exact when s is not small */                             \
        auto rsq = __riscv_vfmul_vv_f##TLEN##m##LMUL(r, r, vl);                \
        auto s_tail = __riscv_vmerge_vvm_f32m##LMUL(zero, s, s_is_small, vl);  \
        /* s_head + s_tail = s; and 1 +/- s is (1 +/- s_head) +/- s_tail */    \
        /* exp(r') is approximated by 1 + r' + (r')^2/2 + r^3(p_even(r^2) +    \
           r*p_odd(r^2)) using r without delta_r sufficies from the third      \
         order onwards */                                                      \
        auto rcube = __riscv_vfmul_vv_f##TLEN##m##LMUL(rsq, r, vl);            \
        auto c0 = __riscv_vfmv_v_f_f32m##LMUL(0x1.71ddef82f4beep-19, vl);      \
        auto c1 = __riscv_vfmv_v_f_f32m##LMUL(0x1.a01a01b32b633p-13, vl);      \
        auto c2 = __riscv_vfmv_v_f_f32m##LMUL(0x1.111111110ef6ap-7, vl);       \
        auto c3 = __riscv_vfmv_v_f_f32m##LMUL(0x1.555555555555ap-3, vl);       \
        auto c4 = __riscv_vfmv_v_f_f32m##LMUL(0x1.a019b37a2b3dfp-16, vl);      \
        auto c5 = __riscv_vfmv_v_f_f32m##LMUL(0x1.6c16c17a09506p-10, vl);      \
        auto c6 = __riscv_vfmv_v_f_f32m##LMUL(0x1.5555555553aefp-5, vl);       \
                                                                               \
        auto p_even = __riscv_vmv_v_v_f32m##LMUL(rsq, vl);                     \
        p_even = __riscv_vfmadd_vf_f32m##LMUL(p_even, 0x1.af6eacd796f0bp-26,   \
                                              c0, vl);                         \
        p_even = __riscv_vfmadd_vv_f32m##LMUL(p_even, rsq, c1, vl);            \
        p_even = __riscv_vfmadd_vv_f32m##LMUL(p_even, rsq, c2, vl);            \
        p_even = __riscv_vfmadd_vv_f32m##LMUL(p_even, rsq, c3, vl);            \
                                                                               \
        auto p_odd = __riscv_vmv_v_v_f32m##LMUL(rsq, vl);                      \
        p_odd = __riscv_vfmadd_vf_f32m##LMUL(p_odd, 0x1.289788d8bdadfp-22, c4, \
                                             vl);                              \
        p_odd = __riscv_vfmadd_vv_f32m##LMUL(p_odd, rsq, c5, vl);              \
        p_odd = __riscv_vfmadd_vv_f32m##LMUL(p_odd, rsq, c6, vl);              \
        auto poly = __riscv_vfmadd_vv_f32m##LMUL(p_odd, r, p_even, vl);        \
                                                                               \
        /* r^3 * poly will be r^3(...)                                         \
           we delay this multiplication with r^3 for now */                    \
                                                                               \
        /*  Compute r' + (r')^2/2 extra precisely */                           \
        auto r_prime = __riscv_vfmul_vf_f32m##LMUL(r, 0x1.0p-1, vl);           \
        auto B = __riscv_vfmadd_vv_f32m##LMUL(r, r_prime, r, vl);              \
        auto b = __riscv_vfsub_vv_f32m##LMUL(r, B, vl);                        \
        b = __riscv_vfmacc_vv_f32m##LMUL(b, r, r_prime, vl);                   \
        /* B + b is r' + (r')^2/2 extra precisely */                           \
        /* incoporate r_delta in R + R^2/2 */                                  \
        auto c = __riscv_vfmadd_vv_f32m##LMUL(r, r_delta, r_delta, vl);        \
        b = __riscv_vfadd_vv_f32m##LMUL(b, c, vl);                             \
        poly = __riscv_vfmadd_vv_f32m##LMUL(poly, rcube, b, vl);               \
        /* B + poly is r' + (r')^2/2 + r^3(.....) */                           \
        /* and exp(r') is well approximated by s*(1 + B + poly) */             \
                                                                               \
        /* We compute the denominator 1 + exp(R) first as                      \
           we will need to recipricate afterwards, the latency of which        \
           can be hidden somewhat by proceeding with the numerator             \
           at that time */                                                     \
        auto Z = __riscv_vfadd_vf_f32m##LMUL(s_head, fp_posOne, vl);           \
        auto D_tmp = __riscv_vfmadd_vv_f32m##LMUL(B, s, Z, vl);                \
        auto d_tmp = __riscv_vfsub_vv_f32m##LMUL(Z, D_tmp, vl);                \
        d_tmp = __riscv_vfmacc_vv_f32m##LMUL(d_tmp, s, B, vl);                 \
        d_tmp = __riscv_vfadd_vv_f32m##LMUL(d_tmp, s_tail, vl);                \
        d_tmp = __riscv_vfmacc_vv_f32m##LMUL(d_tmp, s, poly, vl);              \
        /* D_tmp + d_tmp is 1 + exp(R) to high precision, but we have to       \
           normalize this representation so that the leading term              \
           has full FP64 precision of this sum */                              \
        auto D = __riscv_vfadd_vv_f32m##LMUL(D_tmp, d_tmp, vl);                \
        auto d = __riscv_vfsub_vv_f32m##LMUL(D_tmp, D, vl);                    \
        d = __riscv_vfadd_vv_f32m##LMUL(d, d_tmp, vl);                         \
                                                                               \
        /* Now start to compute 1/(D+d) as E + e */                            \
        auto E = __riscv_vfrdiv_vf_f32m##LMUL(D, fp_posOne, vl);               \
        auto e = __riscv_vfnmsub_vv_f32m##LMUL(E, D, one, vl);                 \
        e = __riscv_vfnmsac_vv_f32m##LMUL(e, E, d, vl);                        \
        e = __riscv_vfmul_vv_f32m##LMUL(e, __riscv_vfrec7_v_f32m##LMUL(D, vl), \
                                        vl);                                   \
        /* E + e is 1/(D+d) to extra precision */                              \
                                                                               \
        /* Overlap much of the 1/(D+d) computation with */                     \
        /* computing 1 - s(1 + B + poly) */                                    \
        Z = __riscv_vfrsub_vf_f32m##LMUL(s_head, fp_posOne, vl);               \
                                                                               \
        auto Numer = __riscv_vfnmsub_vv_f32m##LMUL(B, s, Z, vl);               \
        auto numer = __riscv_vfsub_vv_f32m##LMUL(Z, Numer, vl);                \
        numer = __riscv_vfnmsac_vv_f32m##LMUL(numer, s, B, vl);                \
                                                                               \
        /* Numer + numer = Z - s * B accurately */                             \
        numer = __riscv_vfsub_vv_f32m##LMUL(numer, s_tail, vl);                \
        numer = __riscv_vfnmsac_vv_f32m##LMUL(numer, s, poly, vl);             \
                                                                               \
        /* (Numer + numer) * (E + e) */                                        \
        /* Numer * E + ( numer * E + (Numer * e + (e*numer)) ) */              \
        auto vy = __riscv_vfmul_vv_f32m##LMUL(e, numer, vl);                   \
        vy = __riscv_vfmacc_vv_f32m##LMUL(vy, Numer, e, vl);                   \
        vy = __riscv_vfmacc_vv_f32m##LMUL(vy, numer, E, vl);                   \
        vy = __riscv_vfmacc_vv_f32m##LMUL(vy, Numer, E, vl);                   \
        return __riscv_vfsgnj_vv_f32##m##LMUL(vy, v, vl);                      \
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

#endif