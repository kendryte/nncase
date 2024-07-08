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

#if 0
// tylor 11 terms
#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        x = vfmax_vf_f32m##LMUL(                                               \
            x, 0.f, vl); /* force flush to zero on denormal values */          \
        vbool##MLEN##_t invalid_mask =                                         \
            vmfle_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                         \
                                                                               \
        vint32m##LMUL##_t ux = vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);      \
                                                                               \
        vint32m##LMUL##_t emm0 = vsra_vx_i32m##LMUL(ux, 23, vl);               \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);                      \
        ux = vor_vx_i32m##LMUL(                                                \
            ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);              \
        x = vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                        \
                                                                               \
        emm0 = vsub_vx_i32m##LMUL(emm0, 0x7e, vl);                             \
        auto e = vfcvt_f_x_v_f32m##LMUL(emm0, vl);                             \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##MLEN##_t mask =                                                 \
            vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);             \
        x = vfadd_vv_f32m##LMUL##_m(mask, x, x, x, vl);                        \
        e = vfsub_vf_f32m##LMUL##_m(mask, e, e, 1.f, vl);                      \
        x = vfsub_vf_f32m##LMUL(x, 1.f, vl);                                   \
                                                                               \
        auto y1 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto y2 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto y3 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto y4 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto y5 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto c1 = vfmv_v_f_f32m##LMUL(c_cephes_log_p1, vl);                    \
        auto c3 = vfmv_v_f_f32m##LMUL(c_cephes_log_p3, vl);                    \
        auto c5 = vfmv_v_f_f32m##LMUL(c_cephes_log_p5, vl);                    \
        auto c7 = vfmv_v_f_f32m##LMUL(c_cephes_log_p7, vl);                    \
        auto minus_half = vfmv_v_f_f32m##LMUL(-0.5f, vl);                      \
        auto x2 = vfmul_vv_f32m##LMUL(x, x, vl);                               \
        /*f(x) = x - x^2 / 2 + p8x^3 + ... + p0x^11 + (q1 + q2) * e */         \
        /*     = x + x^2(-1/2 + p8x + ... +  x^2(p1 + p0x)) + (q1 + q2) * e */ \
        y1 = vfmadd_vf_f32m##LMUL(y1, c_cephes_log_p0, c1, vl);                \
        y2 = vfmadd_vf_f32m##LMUL(y2, c_cephes_log_p2, c3, vl);                \
        y3 = vfmadd_vf_f32m##LMUL(y3, c_cephes_log_p4, c5, vl);                \
        y4 = vfmadd_vf_f32m##LMUL(y4, c_cephes_log_p6, c7, vl);                \
        y5 = vfmadd_vf_f32m##LMUL(y5, c_cephes_log_p8, minus_half, vl);        \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, y2, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, y3, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, y4, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, y5, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, x, vl);                              \
        y1 = vfmacc_vf_f32m##LMUL(y1, 0.6931471805599453f, e, vl);             \
                                                                               \
        /* negative arg will be NAN */                                         \
        vuint32m##LMUL##_t xtmp = vreinterpret_v_f32m##LMUL##_u32m##LMUL(y1);  \
        y1 = vreinterpret_v_u32m##LMUL##_f32m##LMUL(                           \
            vor_vx_u32m##LMUL##_m(invalid_mask, xtmp, xtmp, 0xffffffff, vl));  \
        return y1;                                                             \
    }
#else
// tylor 5 terms
#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        x = vfmax_vf_f32m##LMUL(                                               \
            x, 0.f, vl); /* force flush to zero on denormal values */          \
        vbool##MLEN##_t invalid_mask =                                         \
            vmfle_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                         \
                                                                               \
        vint32m##LMUL##_t ux = vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);      \
                                                                               \
        vint32m##LMUL##_t emm0 = vsra_vx_i32m##LMUL(ux, 23, vl);               \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);                      \
        ux = vor_vx_i32m##LMUL(                                                \
            ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);              \
        x = vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                        \
                                                                               \
        emm0 = vsub_vx_i32m##LMUL(emm0, 0x7e, vl);                             \
        auto e = vfcvt_f_x_v_f32m##LMUL(emm0, vl);                             \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##MLEN##_t mask =                                                 \
            vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);             \
        x = vfadd_vv_f32m##LMUL##_m(mask, x, x, x, vl);                        \
        e = vfsub_vf_f32m##LMUL##_m(mask, e, e, 1.f, vl);                      \
        x = vfsub_vf_f32m##LMUL(x, 1.f, vl);                                   \
                                                                               \
        auto y1 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto y2 = vmv_v_v_f32m##LMUL(x, vl);                                   \
        auto c7 = vfmv_v_f_f32m##LMUL(c_cephes_log_p7, vl);                    \
        auto minus_half = vfmv_v_f_f32m##LMUL(-0.5f, vl);                      \
        auto x2 = vfmul_vv_f32m##LMUL(x, x, vl);                               \
        /*f(x) = x - x^2 / 2 + p8x^3 + ... + p6x^5 + (q1 + q2) * e */          \
        /*     = x + x^2(-1/2 + p8x + ... +  x^2(p7 + p6x)) + (q1 + q2) * e */ \
        y1 = vfmadd_vf_f32m##LMUL(y1, c_cephes_log_p6, c7, vl);                \
        y2 = vfmadd_vf_f32m##LMUL(y2, c_cephes_log_p8, minus_half, vl);        \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, y2, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, x2, x, vl);                              \
        y1 = vfmacc_vf_f32m##LMUL(y1, 0.6931471805599453f, e, vl);             \
                                                                               \
        /* negative arg will be NAN */                                         \
        vuint32m##LMUL##_t xtmp = vreinterpret_v_f32m##LMUL##_u32m##LMUL(y1);  \
        y1 = vreinterpret_v_u32m##LMUL##_f32m##LMUL(                           \
            vor_vx_u32m##LMUL##_m(invalid_mask, xtmp, xtmp, 0xffffffff, vl));  \
        return y1;                                                             \
    }
#endif

_RVV_FLOAT32_LOG_OP(1, 32)
_RVV_FLOAT32_LOG_OP(2, 16)
_RVV_FLOAT32_LOG_OP(4, 8)
_RVV_FLOAT32_LOG_OP(8, 4)

#define c_inv_mant_fp16_mask (uint16_t) ~0x7c00u
#define _RVV_FLOAT16_LOG_OP(LMUL, MLEN, TLEN)                                  \
    static inline vfloat##TLEN##m##LMUL##_t log_ps(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        x = vfmax_vf_f##TLEN##m##LMUL(                                         \
            x, 0.f, vl); /* force flush to zero on denormal values */          \
        vbool##MLEN##_t invalid_mask =                                         \
            vmfle_vf_f##TLEN##m##LMUL##_b##MLEN(x, 0.f, vl);                   \
                                                                               \
        vint##TLEN##m##LMUL##_t ux =                                           \
            vreinterpret_v_f##TLEN##m##LMUL##_i##TLEN##m##LMUL(x);             \
                                                                               \
        vint##TLEN##m##LMUL##_t emm0 = vsra_vx_i##TLEN##m##LMUL(ux, 10, vl);   \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = vand_vx_i##TLEN##m##LMUL(ux, c_inv_mant_fp16_mask, vl);           \
        ux = vor_vx_i##TLEN##m##LMUL(                                          \
            ux, 14336 /* reinterpret_cast<int>(0.5) */, vl);                   \
        x = vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(ux);            \
                                                                               \
        emm0 = vsub_vx_i##TLEN##m##LMUL(emm0, 0xf, vl);                        \
        vfloat##TLEN##m##LMUL##_t e = vfcvt_f_x_v_f##TLEN##m##LMUL(emm0, vl);  \
                                                                               \
        e = vfadd_vf_f##TLEN##m##LMUL(e, 1.f, vl);                             \
                                                                               \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##MLEN##_t mask =                                                 \
            vmflt_vf_f##TLEN##m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);       \
        x = vfadd_vv_f##TLEN##m##LMUL##_m(mask, x, x, x, vl);                  \
        x = vfsub_vf_f##TLEN##m##LMUL(x, 1.f, vl);                             \
        e = vfsub_vf_f##TLEN##m##LMUL##_m(mask, e, e, 1.f, vl);                \
                                                                               \
        vfloat##TLEN##m##LMUL##_t z = vfmul_vv_f##TLEN##m##LMUL(x, x, vl);     \
                                                                               \
        vfloat##TLEN##m##LMUL##_t y =                                          \
            vfmul_vf_f##TLEN##m##LMUL(x, c_cephes_log_p0, vl);                 \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p1, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p2, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p3, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p4, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p5, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p6, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p7, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
        y = vfadd_vf_f##TLEN##m##LMUL(y, c_cephes_log_p8, vl);                 \
        y = vfmul_vv_f##TLEN##m##LMUL(y, x, vl);                               \
                                                                               \
        y = vfmul_vv_f##TLEN##m##LMUL(y, z, vl);                               \
                                                                               \
        vfloat##TLEN##m##LMUL##_t tmp =                                        \
            vfmul_vf_f##TLEN##m##LMUL(e, c_cephes_log_q1, vl);                 \
        y = vfadd_vv_f##TLEN##m##LMUL(y, tmp, vl);                             \
                                                                               \
        tmp = vfmul_vf_f##TLEN##m##LMUL(z, 0.5f, vl);                          \
        y = vfsub_vv_f##TLEN##m##LMUL(y, tmp, vl);                             \
                                                                               \
        tmp = vfmul_vf_f##TLEN##m##LMUL(e, c_cephes_log_q2, vl);               \
        x = vfadd_vv_f##TLEN##m##LMUL(x, y, vl);                               \
        x = vfadd_vv_f##TLEN##m##LMUL(x, tmp, vl);                             \
        /* negative arg will be NAN */                                         \
        vuint##TLEN##m##LMUL##_t xtmp =                                        \
            vreinterpret_v_f##TLEN##m##LMUL##_u##TLEN##m##LMUL(x);             \
        x = vreinterpret_v_u##TLEN##m##LMUL##_f##TLEN##m##LMUL(                \
            vor_vx_u##TLEN##m##LMUL##_m(invalid_mask, xtmp, xtmp, 0x7e00,      \
                                        vl));                                  \
        return x;                                                              \
    }

_RVV_FLOAT16_LOG_OP(1, 16, 16)
_RVV_FLOAT16_LOG_OP(2, 8, 16)
_RVV_FLOAT16_LOG_OP(4, 4, 16)
_RVV_FLOAT16_LOG_OP(8, 2, 16)

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

// exp(x) = 1 + x + x^2/2! + x^3 / 3! + x^4 / 4!
#if 0
#define _RVV_FLOAT_EXP_OP(LMUL, MLEN, TLEN, E, M)                              \
    static inline vfloat##TLEN##m##LMUL##_t exp_ps(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        x = vfmin_vf_f##TLEN##m##LMUL(x, c_exp_hi, vl);                        \
        x = vfmax_vf_f##TLEN##m##LMUL(x, c_exp_lo, vl);                        \
        auto y = vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p4, vl);               \
        auto c1 = vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p5, vl);              \
        auto c2 = vfmv_v_f_f##TLEN##m##LMUL(1.f, vl);                          \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        auto fx = vfmacc_vf_f##TLEN##m##LMUL(c1, c_cephes_LOG2EF, x, vl);      \
                                                                               \
        /* perform a floorf */                                                 \
        auto tmp = vfcvt_f_x_v_f##TLEN##m##LMUL(                               \
            vfcvt_x_f_v_i##TLEN##m##LMUL(fx, vl), vl);                         \
                                                                               \
        /* if greater, substract 1 */                                          \
        auto mask = vmfgt_vv_f##TLEN##m##LMUL##_b##MLEN(tmp, fx, vl);          \
        fx = vfsub_vf_f##TLEN##m##LMUL##_m(mask, tmp, tmp, 1.f, vl);           \
                                                                               \
        x = vfmacc_vf_f##TLEN##m##LMUL(x, -c_cephes_exp_C1, fx, vl);           \
                                                                               \
        y = vfmacc_vf_f##TLEN##m##LMUL(y, c_cephes_exp_p3, x, vl);             \
        y = vfmadd_vv_f##TLEN##m##LMUL(y, x, c1, vl);                          \
        y = vfmadd_vv_f##TLEN##m##LMUL(y, x, c2, vl);                          \
        y = vfmadd_vv_f##TLEN##m##LMUL(y, x, c2, vl);                          \
        auto b = vreinterpret_v_f##TLEN##m##LMUL##_i##TLEN##m##LMUL(y);        \
                                                                               \
        /* build 2^n */                                                        \
        auto a = vsll_vx_i##TLEN##m##LMUL(                                     \
            vfcvt_x_f_v_i##TLEN##m##LMUL(fx, vl), M, vl);                      \
        auto ret = vadd_vv_i##TLEN##m##LMUL(a, b, vl);                         \
        return vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(ret);        \
    }
#else
// exp(x) = 1 + x(1 + 1/6 * x^2)  x^2(1/2 + 1/24 * x^2)
//        = 1 + x + x^2(1/2 + 1/6 * x + 1/24 * x^2)
#define _RVV_FLOAT_EXP_OP(LMUL, MLEN, TLEN, E, M)                              \
    static inline vfloat##TLEN##m##LMUL##_t exp_ps(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        x = vfmin_vf_f##TLEN##m##LMUL(x, c_exp_hi, vl);                        \
        x = vfmax_vf_f##TLEN##m##LMUL(x, c_exp_lo, vl);                        \
        auto c1 = vfmv_v_f_f##TLEN##m##LMUL(c_cephes_exp_p5, vl);              \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        auto fx = vfmacc_vf_f##TLEN##m##LMUL(c1, c_cephes_LOG2EF, x, vl);      \
                                                                               \
        /* perform a floorf */                                                 \
        auto tmp = vfcvt_f_x_v_f##TLEN##m##LMUL(                               \
            vfcvt_x_f_v_i##TLEN##m##LMUL(fx, vl), vl);                         \
                                                                               \
        /* if greater, substract 1 */                                          \
        auto mask = vmfgt_vv_f##TLEN##m##LMUL##_b##MLEN(tmp, fx, vl);          \
        fx = vfsub_vf_f##TLEN##m##LMUL##_m(mask, tmp, tmp, 1.f, vl);           \
        x = vfmacc_vf_f##TLEN##m##LMUL(x, -c_cephes_exp_C1, fx, vl);           \
                                                                               \
        auto y1 = vfmv_v_f_f##TLEN##m##LMUL(0.5f, vl);                         \
        y1 = vfmacc_vf_f##TLEN##m##LMUL(y1, c_cephes_exp_p4, x, vl);           \
        auto x2 = vfmul_vv_f##TLEN##m##LMUL(x, x, vl);                         \
        auto y2 = vfadd_vf_f##TLEN##m##LMUL(x, 1.f, vl);                       \
        y1 = vfmacc_vf_f##TLEN##m##LMUL(y1, c_cephes_exp_p3, x2, vl);          \
        y1 = vfmadd_vv_f##TLEN##m##LMUL(y1, x2, y2, vl);                       \
        auto b = vreinterpret_v_f##TLEN##m##LMUL##_i##TLEN##m##LMUL(y1);       \
                                                                               \
        /* build 2^n */                                                        \
        auto a = vsll_vx_i##TLEN##m##LMUL(                                     \
            vfcvt_x_f_v_i##TLEN##m##LMUL(fx, vl), M, vl);                      \
        auto ret = vadd_vv_i##TLEN##m##LMUL(a, b, vl);                         \
        return vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(ret);        \
    }
#endif
_RVV_FLOAT_EXP_OP(1, 32, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(2, 16, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(4, 8, 32, 0x7f, 23)
_RVV_FLOAT_EXP_OP(8, 4, 32, 0x7f, 23)

_RVV_FLOAT_EXP_OP(1, 16, 16, 0xf, 10)
_RVV_FLOAT_EXP_OP(2, 8, 16, 0xf, 10)
_RVV_FLOAT_EXP_OP(4, 4, 16, 0xf, 10)
_RVV_FLOAT_EXP_OP(8, 2, 16, 0xf, 10)

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

#define _RVV_FLOAT_SINCOS_OP(LMUL, MLEN, TLEN)                                 \
    static inline void sincos_ps(vfloat##TLEN##m##LMUL##_t x,                  \
                                 vfloat##TLEN##m##LMUL##_t *ysin,              \
                                 vfloat##TLEN##m##LMUL##_t *ycos, size_t vl) { \
        /* any x */                                                            \
        vfloat##TLEN##m##LMUL##_t xmm1, xmm2, xmm3, y;                         \
                                                                               \
        vuint##TLEN##m##LMUL##_t emm2;                                         \
                                                                               \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                          \
        sign_mask_sin = vmflt_vf_f##TLEN##m##LMUL##_b##MLEN(x, 0.f, vl);       \
        x = vfsgnj_vf_f##TLEN##m##LMUL(x, 1.f, vl);                            \
                                                                               \
        /* scale by 4/Pi */                                                    \
        y = vfmul_vf_f##TLEN##m##LMUL(x, c_cephes_FOPI, vl);                   \
                                                                               \
        /* store the integer part of y in mm0 */                               \
        emm2 = vfcvt_xu_f_v_u##TLEN##m##LMUL(y, vl);                           \
        /* j=(j+1) & (~1) (see the cephes sources) */                          \
        emm2 = vadd_vx_u##TLEN##m##LMUL(emm2, 1, vl);                          \
        emm2 = vand_vx_u##TLEN##m##LMUL(emm2, ~1, vl);                         \
        y = vfcvt_f_xu_v_f##TLEN##m##LMUL(emm2, vl);                           \
                                                                               \
        /* get the polynom selection mask              */                      \
        /*     there is one polynom for 0 <= x <= Pi/4 */                      \
        /*     and another one for Pi/4<x<=Pi/2        */                      \
        /*                                             */                      \
        /*     Both branches will be computed.         */                      \
        vbool##MLEN##_t poly_mask = vmsne_vx_u##TLEN##m##LMUL##_b##MLEN(       \
            vand_vx_u##TLEN##m##LMUL(emm2, 2, vl), 0, vl);                     \
                                                                               \
        /* The magic pass: "Extended precision modular arithmetic" */          \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */          \
        xmm1 = vfmul_vf_f##TLEN##m##LMUL(y, c_minus_cephes_DP1, vl);           \
        xmm2 = vfmul_vf_f##TLEN##m##LMUL(y, c_minus_cephes_DP2, vl);           \
        xmm3 = vfmul_vf_f##TLEN##m##LMUL(y, c_minus_cephes_DP3, vl);           \
        x = vfadd_vv_f##TLEN##m##LMUL(x, xmm1, vl);                            \
        x = vfadd_vv_f##TLEN##m##LMUL(x, xmm2, vl);                            \
        x = vfadd_vv_f##TLEN##m##LMUL(x, xmm3, vl);                            \
                                                                               \
        sign_mask_sin = vmxor_mm_b##MLEN(                                      \
            sign_mask_sin,                                                     \
            vmsne_vx_u##TLEN##m##LMUL##_b##MLEN(                               \
                vand_vx_u##TLEN##m##LMUL(emm2, 4, vl), 0, vl),                 \
            vl);                                                               \
        sign_mask_cos = vmsne_vx_u##TLEN##m##LMUL##_b##MLEN(                   \
            vand_vx_u##TLEN##m##LMUL(vsub_vx_u##TLEN##m##LMUL(emm2, 2, vl), 4, \
                                     vl),                                      \
            0, vl);                                                            \
                                                                               \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */              \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */              \
        vfloat##TLEN##m##LMUL##_t z = vfmul_vv_f##TLEN##m##LMUL(x, x, vl);     \
        vfloat##TLEN##m##LMUL##_t y1, y2;                                      \
                                                                               \
        y1 = vfmul_vf_f##TLEN##m##LMUL(z, c_coscof_p0, vl);                    \
        y2 = vfmul_vf_f##TLEN##m##LMUL(z, c_sincof_p0, vl);                    \
        y1 = vfadd_vf_f##TLEN##m##LMUL(y1, c_coscof_p1, vl);                   \
        y2 = vfadd_vf_f##TLEN##m##LMUL(y2, c_sincof_p1, vl);                   \
        y1 = vfmul_vv_f##TLEN##m##LMUL(y1, z, vl);                             \
        y2 = vfmul_vv_f##TLEN##m##LMUL(y2, z, vl);                             \
        y1 = vfadd_vf_f##TLEN##m##LMUL(y1, c_coscof_p2, vl);                   \
        y2 = vfadd_vf_f##TLEN##m##LMUL(y2, c_sincof_p2, vl);                   \
        y1 = vfmul_vv_f##TLEN##m##LMUL(y1, z, vl);                             \
        y2 = vfmul_vv_f##TLEN##m##LMUL(y2, z, vl);                             \
        y1 = vfmul_vv_f##TLEN##m##LMUL(y1, z, vl);                             \
        y2 = vfmul_vv_f##TLEN##m##LMUL(y2, x, vl);                             \
        y1 = vfsub_vv_f##TLEN##m##LMUL(                                        \
            y1, vfmul_vf_f##TLEN##m##LMUL(z, 0.5f, vl), vl);                   \
        y2 = vfadd_vv_f##TLEN##m##LMUL(y2, x, vl);                             \
        y1 = vfadd_vf_f##TLEN##m##LMUL(y1, 1.f, vl);                           \
                                                                               \
        /* select the correct result from the two polynoms */                  \
        vfloat##TLEN##m##LMUL##_t ys =                                         \
            vmerge_vvm_f##TLEN##m##LMUL(poly_mask, y2, y1, vl);                \
        vfloat##TLEN##m##LMUL##_t yc =                                         \
            vmerge_vvm_f##TLEN##m##LMUL(poly_mask, y1, y2, vl);                \
        *ysin = vmerge_vvm_f##TLEN##m##LMUL(                                   \
            sign_mask_sin, ys, vfneg_v_f##TLEN##m##LMUL(ys, vl), vl);          \
        *ycos = vmerge_vvm_f##TLEN##m##LMUL(                                   \
            sign_mask_cos, vfneg_v_f##TLEN##m##LMUL(yc, vl), yc, vl);          \
    }

_RVV_FLOAT_SINCOS_OP(1, 32, 32)
_RVV_FLOAT_SINCOS_OP(2, 16, 32)
_RVV_FLOAT_SINCOS_OP(4, 8, 32)
_RVV_FLOAT_SINCOS_OP(8, 4, 32)

_RVV_FLOAT_SINCOS_OP(1, 16, 16)
_RVV_FLOAT_SINCOS_OP(2, 8, 16)
_RVV_FLOAT_SINCOS_OP(4, 4, 16)
_RVV_FLOAT_SINCOS_OP(8, 2, 16)

#define _RVV_FLOAT_SIN_OP(LMUL, MLEN, TLEN)                                    \
    static inline vfloat##TLEN##m##LMUL##_t sin_ps(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        vfloat##TLEN##m##LMUL##_t ysin, ycos;                                  \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ysin;                                                           \
    }

_RVV_FLOAT_SIN_OP(1, 32, 32)
_RVV_FLOAT_SIN_OP(2, 16, 32)
_RVV_FLOAT_SIN_OP(4, 8, 32)
_RVV_FLOAT_SIN_OP(8, 4, 32)

_RVV_FLOAT_SIN_OP(1, 16, 16)
_RVV_FLOAT_SIN_OP(2, 32, 16)
_RVV_FLOAT_SIN_OP(4, 4, 16)
_RVV_FLOAT_SIN_OP(8, 2, 16)

#define _RVV_FLOAT_COS_OP(LMUL, MLEN, TLEN)                                    \
    static inline vfloat##TLEN##m##LMUL##_t cos_ps(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        vfloat##TLEN##m##LMUL##_t ysin, ycos;                                  \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ycos;                                                           \
    }

_RVV_FLOAT_COS_OP(1, 32, 32)
_RVV_FLOAT_COS_OP(2, 16, 32)
_RVV_FLOAT_COS_OP(4, 8, 32)
_RVV_FLOAT_COS_OP(8, 4, 32)

_RVV_FLOAT_COS_OP(1, 16, 16)
_RVV_FLOAT_COS_OP(2, 8, 16)
_RVV_FLOAT_COS_OP(4, 4, 16)
_RVV_FLOAT_COS_OP(8, 2, 16)

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
        auto abs = vfabs_v_f##TLEN##m##LMUL(x, vl);                            \
                                                                               \
        /* clamp the inputs to the range [-9, 9] since anything outside */     \
        /* this range is -/+1.0f in single-precision.                   */     \
        abs = vfmin_vf_f##TLEN##m##LMUL(abs, c_tanh_hi, vl);                   \
                                                                               \
        /* since the polynomials are odd/even, we need x**2. */                \
        auto x2 = vfmul_vv_f##TLEN##m##LMUL(abs, abs, vl);                     \
                                                                               \
        /* evaluate the numerator polynomial y, denominator polynomial w. */   \
        auto c0 = vfmv_v_f_f##TLEN##m##LMUL(c_tanh_beta_0, vl);                \
        auto c1 = vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_1, vl);               \
        auto c4 = vfmv_v_f_f##TLEN##m##LMUL(c_tanh_beta_4, vl);                \
        auto c5 = vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_5, vl);               \
        auto c9 = vfmv_v_f_f##TLEN##m##LMUL(c_tanh_alpha_9, vl);               \
        auto y1 = vmv_v_v_f##TLEN##m##LMUL(x2, vl);                            \
        auto y2 = vmv_v_v_f##TLEN##m##LMUL(x2, vl);                            \
        auto y3 = vmv_v_v_f##TLEN##m##LMUL(x2, vl);                            \
        auto w1 = vmv_v_v_f##TLEN##m##LMUL(x2, vl);                            \
        auto w2 = vmv_v_v_f##TLEN##m##LMUL(x2, vl);                            \
        y1 = vfmadd_vf_f##TLEN##m##LMUL(y1, c_tanh_alpha_11, c9, vl);          \
        w1 = vfmadd_vf_f##TLEN##m##LMUL(w1, c_tanh_beta_6, c4, vl);            \
        auto x4 = vfmul_vv_f##TLEN##m##LMUL(x2, x2, vl);                       \
        y1 = vfmacc_vf_f##TLEN##m##LMUL(y1, c_tanh_alpha_13, x4, vl);          \
        y2 = vfmadd_vf_f##TLEN##m##LMUL(y2, c_tanh_alpha_7, c5, vl);           \
        y1 = vfmadd_vv_f##TLEN##m##LMUL(y1, x4, y2, vl);                       \
        w2 = vfmadd_vf_f##TLEN##m##LMUL(w2, c_tanh_beta_2, c0, vl);            \
        y3 = vfmadd_vf_f##TLEN##m##LMUL(y3, c_tanh_alpha_3, c1, vl);           \
        auto w = vfmadd_vv_f##TLEN##m##LMUL(w1, x4, w2, vl);                   \
        y1 = vfmadd_vv_f##TLEN##m##LMUL(y1, x4, y3, vl);                       \
        auto z = vfsgnj_vv_f##TLEN##m##LMUL(abs, x, vl);                       \
        w = vfrec7_v_f##TLEN##m##LMUL(w, vl);                                  \
        y1 = vfmul_vv_f##TLEN##m##LMUL(y1, z, vl);                             \
        auto tiny_mask =                                                       \
            vmfge_vf_f##TLEN##m##LMUL##_b##MLEN(abs, c_tanh_tiny, vl);         \
                                                                               \
        /* divide the numerator by the denominator. */                         \
        auto y = vfmul_vv_f##TLEN##m##LMUL(y1, w, vl);                         \
                                                                               \
        /* when the argument is very small in magnitude it's more accurate to  \
         * just return it. */                                                  \
        y = vmerge_vvm_f##TLEN##m##LMUL(tiny_mask, x, y, vl);                  \
                                                                               \
        return y;                                                              \
    }

_RVV_FLOAT_TANH_OP(1, 32, 32)
_RVV_FLOAT_TANH_OP(2, 16, 32)
_RVV_FLOAT_TANH_OP(4, 8, 32)
_RVV_FLOAT_TANH_OP(8, 4, 32)

_RVV_FLOAT_TANH_OP(1, 16, 16)
_RVV_FLOAT_TANH_OP(2, 8, 16)
_RVV_FLOAT_TANH_OP(4, 4, 16)
_RVV_FLOAT_TANH_OP(8, 2, 16)

#define _RVV_FLOAT_POW_OP(LMUL, MLEN, TLEN)                                    \
    static inline vfloat##TLEN##m##LMUL##_t pow_ps(                            \
        vfloat##TLEN##m##LMUL##_t a, vfloat##TLEN##m##LMUL##_t b, size_t vl) { \
        /* pow(x, m) = exp(m * log(x)) */                                      \
        return exp_ps(vfmul_vv_f##TLEN##m##LMUL(b, log_ps(a, vl), vl), vl);    \
    }

_RVV_FLOAT_POW_OP(1, 32, 32)
_RVV_FLOAT_POW_OP(2, 16, 32)
_RVV_FLOAT_POW_OP(4, 8, 32)
_RVV_FLOAT_POW_OP(8, 4, 32)

_RVV_FLOAT_POW_OP(1, 16, 16)
_RVV_FLOAT_POW_OP(2, 8, 16)
_RVV_FLOAT_POW_OP(4, 4, 16)
_RVV_FLOAT_POW_OP(8, 2, 16)

#define _RVV_FLOAT_VF_OP(OP, LMUL, TLEN)                                       \
    static inline vfloat##TLEN##m##LMUL##_t vf_##OP##_op(                      \
        vfloat##TLEN##m##LMUL##_t x, float##TLEN##_t y, size_t vl) {           \
        return vf##OP##_vf_f##TLEN##m##LMUL(x, y, vl);                         \
    }

_RVV_FLOAT_VF_OP(add, 8, 32);
_RVV_FLOAT_VF_OP(add, 8, 16);
_RVV_FLOAT_VF_OP(add, 4, 32);
_RVV_FLOAT_VF_OP(add, 4, 16);
_RVV_FLOAT_VF_OP(add, 2, 32);
_RVV_FLOAT_VF_OP(add, 2, 16);
_RVV_FLOAT_VF_OP(add, 1, 32);
_RVV_FLOAT_VF_OP(add, 1, 16);

_RVV_FLOAT_VF_OP(sub, 8, 32);
_RVV_FLOAT_VF_OP(sub, 8, 16);
_RVV_FLOAT_VF_OP(sub, 4, 32);
_RVV_FLOAT_VF_OP(sub, 4, 16);
_RVV_FLOAT_VF_OP(sub, 2, 32);
_RVV_FLOAT_VF_OP(sub, 2, 16);
_RVV_FLOAT_VF_OP(sub, 1, 32);
_RVV_FLOAT_VF_OP(sub, 1, 16);

_RVV_FLOAT_VF_OP(mul, 8, 32);
_RVV_FLOAT_VF_OP(mul, 8, 16);
_RVV_FLOAT_VF_OP(mul, 4, 32);
_RVV_FLOAT_VF_OP(mul, 4, 16);
_RVV_FLOAT_VF_OP(mul, 2, 32);
_RVV_FLOAT_VF_OP(mul, 2, 16);
_RVV_FLOAT_VF_OP(mul, 1, 32);
_RVV_FLOAT_VF_OP(mul, 1, 16);

_RVV_FLOAT_VF_OP(div, 8, 32);
_RVV_FLOAT_VF_OP(div, 8, 16);
_RVV_FLOAT_VF_OP(div, 4, 32);
_RVV_FLOAT_VF_OP(div, 4, 16);
_RVV_FLOAT_VF_OP(div, 2, 32);
_RVV_FLOAT_VF_OP(div, 2, 16);
_RVV_FLOAT_VF_OP(div, 1, 32);
_RVV_FLOAT_VF_OP(div, 1, 16);

_RVV_FLOAT_VF_OP(min, 8, 32);
_RVV_FLOAT_VF_OP(min, 8, 16);
_RVV_FLOAT_VF_OP(min, 4, 32);
_RVV_FLOAT_VF_OP(min, 4, 16);
_RVV_FLOAT_VF_OP(min, 2, 32);
_RVV_FLOAT_VF_OP(min, 2, 16);
_RVV_FLOAT_VF_OP(min, 1, 32);
_RVV_FLOAT_VF_OP(min, 1, 16);

_RVV_FLOAT_VF_OP(max, 8, 32);
_RVV_FLOAT_VF_OP(max, 8, 16);
_RVV_FLOAT_VF_OP(max, 4, 32);
_RVV_FLOAT_VF_OP(max, 4, 16);
_RVV_FLOAT_VF_OP(max, 2, 32);
_RVV_FLOAT_VF_OP(max, 2, 16);
_RVV_FLOAT_VF_OP(max, 1, 32);
_RVV_FLOAT_VF_OP(max, 1, 16);

#define _RVV_FLOAT_FV_OP(OP, LMUL, TLEN)                                       \
    static inline vfloat##TLEN##m##LMUL##_t fv_##OP##_op(                      \
        vfloat##TLEN##m##LMUL##_t x, float##TLEN##_t y, size_t vl) {           \
        return vf##OP##_vf_f##TLEN##m##LMUL(x, y, vl);                         \
    }

#define _RVV_FLOAT_FV_SUB(OP, LMUL, TLEN)                                      \
    static inline vfloat##TLEN##m##LMUL##_t fv_##OP##_op(                      \
        vfloat##TLEN##m##LMUL##_t x, float##TLEN##_t y, size_t vl) {           \
        x = vf##OP##_vf_f##TLEN##m##LMUL(x, y, vl);                            \
        return vfneg_v_f##TLEN##m##LMUL(x, vl);                                \
    }

_RVV_FLOAT_FV_OP(add, 8, 32);
_RVV_FLOAT_FV_OP(add, 8, 16);
_RVV_FLOAT_FV_OP(add, 4, 32);
_RVV_FLOAT_FV_OP(add, 4, 16);
_RVV_FLOAT_FV_OP(add, 2, 32);
_RVV_FLOAT_FV_OP(add, 2, 16);
_RVV_FLOAT_FV_OP(add, 1, 32);
_RVV_FLOAT_FV_OP(add, 1, 16);

_RVV_FLOAT_FV_SUB(sub, 8, 32);
_RVV_FLOAT_FV_SUB(sub, 8, 16);
_RVV_FLOAT_FV_SUB(sub, 4, 32);
_RVV_FLOAT_FV_SUB(sub, 4, 16);
_RVV_FLOAT_FV_SUB(sub, 2, 32);
_RVV_FLOAT_FV_SUB(sub, 2, 16);
_RVV_FLOAT_FV_SUB(sub, 1, 32);
_RVV_FLOAT_FV_SUB(sub, 1, 16);

_RVV_FLOAT_FV_OP(mul, 8, 32);
_RVV_FLOAT_FV_OP(mul, 8, 16);
_RVV_FLOAT_FV_OP(mul, 4, 32);
_RVV_FLOAT_FV_OP(mul, 4, 16);
_RVV_FLOAT_FV_OP(mul, 2, 32);
_RVV_FLOAT_FV_OP(mul, 2, 16);
_RVV_FLOAT_FV_OP(mul, 1, 32);
_RVV_FLOAT_FV_OP(mul, 1, 16);

_RVV_FLOAT_FV_OP(div, 8, 32);
_RVV_FLOAT_FV_OP(div, 8, 16);
_RVV_FLOAT_FV_OP(div, 4, 32);
_RVV_FLOAT_FV_OP(div, 4, 16);
_RVV_FLOAT_FV_OP(div, 2, 32);
_RVV_FLOAT_FV_OP(div, 2, 16);
_RVV_FLOAT_FV_OP(div, 1, 32);
_RVV_FLOAT_FV_OP(div, 1, 16);

_RVV_FLOAT_FV_OP(min, 8, 32);
_RVV_FLOAT_FV_OP(min, 8, 16);
_RVV_FLOAT_FV_OP(min, 4, 32);
_RVV_FLOAT_FV_OP(min, 4, 16);
_RVV_FLOAT_FV_OP(min, 2, 32);
_RVV_FLOAT_FV_OP(min, 2, 16);
_RVV_FLOAT_FV_OP(min, 1, 32);
_RVV_FLOAT_FV_OP(min, 1, 16);

_RVV_FLOAT_FV_OP(max, 8, 32);
_RVV_FLOAT_FV_OP(max, 8, 16);
_RVV_FLOAT_FV_OP(max, 4, 32);
_RVV_FLOAT_FV_OP(max, 4, 16);
_RVV_FLOAT_FV_OP(max, 2, 32);
_RVV_FLOAT_FV_OP(max, 2, 16);
_RVV_FLOAT_FV_OP(max, 1, 32);
_RVV_FLOAT_FV_OP(max, 1, 16);

#define _RVV_FLOAT_VV_OP(OP, LMUL, TLEN)                                       \
    static inline vfloat##TLEN##m##LMUL##_t vv_##OP##_op(                      \
        vfloat##TLEN##m##LMUL##_t x, vfloat##TLEN##m##LMUL##_t y, size_t vl) { \
        return vf##OP##_vv_f##TLEN##m##LMUL(x, y, vl);                         \
    }

_RVV_FLOAT_VV_OP(add, 8, 32);
_RVV_FLOAT_VV_OP(add, 8, 16);
_RVV_FLOAT_VV_OP(add, 4, 32);
_RVV_FLOAT_VV_OP(add, 4, 16);
_RVV_FLOAT_VV_OP(add, 2, 32);
_RVV_FLOAT_VV_OP(add, 2, 16);
_RVV_FLOAT_VV_OP(add, 1, 32);
_RVV_FLOAT_VV_OP(add, 1, 16);

_RVV_FLOAT_VV_OP(sub, 8, 32);
_RVV_FLOAT_VV_OP(sub, 8, 16);
_RVV_FLOAT_VV_OP(sub, 4, 32);
_RVV_FLOAT_VV_OP(sub, 4, 16);
_RVV_FLOAT_VV_OP(sub, 2, 32);
_RVV_FLOAT_VV_OP(sub, 2, 16);
_RVV_FLOAT_VV_OP(sub, 1, 32);
_RVV_FLOAT_VV_OP(sub, 1, 16);

_RVV_FLOAT_VV_OP(mul, 8, 32);
_RVV_FLOAT_VV_OP(mul, 8, 16);
_RVV_FLOAT_VV_OP(mul, 4, 32);
_RVV_FLOAT_VV_OP(mul, 4, 16);
_RVV_FLOAT_VV_OP(mul, 2, 32);
_RVV_FLOAT_VV_OP(mul, 2, 16);
_RVV_FLOAT_VV_OP(mul, 1, 32);
_RVV_FLOAT_VV_OP(mul, 1, 16);

_RVV_FLOAT_VV_OP(div, 8, 32);
_RVV_FLOAT_VV_OP(div, 8, 16);
_RVV_FLOAT_VV_OP(div, 4, 32);
_RVV_FLOAT_VV_OP(div, 4, 16);
_RVV_FLOAT_VV_OP(div, 2, 32);
_RVV_FLOAT_VV_OP(div, 2, 16);
_RVV_FLOAT_VV_OP(div, 1, 32);
_RVV_FLOAT_VV_OP(div, 1, 16);

_RVV_FLOAT_VV_OP(min, 8, 32);
_RVV_FLOAT_VV_OP(min, 8, 16);
_RVV_FLOAT_VV_OP(min, 4, 32);
_RVV_FLOAT_VV_OP(min, 4, 16);
_RVV_FLOAT_VV_OP(min, 2, 32);
_RVV_FLOAT_VV_OP(min, 2, 16);
_RVV_FLOAT_VV_OP(min, 1, 32);
_RVV_FLOAT_VV_OP(min, 1, 16);

_RVV_FLOAT_VV_OP(max, 8, 32);
_RVV_FLOAT_VV_OP(max, 8, 16);
_RVV_FLOAT_VV_OP(max, 4, 32);
_RVV_FLOAT_VV_OP(max, 4, 16);
_RVV_FLOAT_VV_OP(max, 2, 32);
_RVV_FLOAT_VV_OP(max, 2, 16);
_RVV_FLOAT_VV_OP(max, 1, 32);
_RVV_FLOAT_VV_OP(max, 1, 16);

#define _RVV_FLOAT_ABS_OP(LMUL, TLEN)                                          \
    static inline vfloat##TLEN##m##LMUL##_t abs_op(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfabs_v_f##TLEN##m##LMUL(x, vl);                                \
    }

_RVV_FLOAT_ABS_OP(8, 32)
_RVV_FLOAT_ABS_OP(4, 32)
_RVV_FLOAT_ABS_OP(2, 32)
_RVV_FLOAT_ABS_OP(1, 32)
_RVV_FLOAT_ABS_OP(8, 16)
_RVV_FLOAT_ABS_OP(4, 16)
_RVV_FLOAT_ABS_OP(2, 16)
_RVV_FLOAT_ABS_OP(1, 16)

#define _RVV_FLOAT_CEIL_OP(LMUL, MLEN, TLEN)                                   \
    static inline vfloat##TLEN##m##LMUL##_t ceil_op(                           \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        vint##TLEN##m##LMUL##_t _xi = vfcvt_x_f_v_i##TLEN##m##LMUL(x, vl);     \
        auto _mask = vmflt_vv_f##TLEN##m##LMUL##_b##MLEN(                      \
            vfcvt_f_x_v_f##TLEN##m##LMUL(_xi, vl), x, vl);                     \
        return vfcvt_f_x_v_f##TLEN##m##LMUL(                                   \
            vadd_vx_i##TLEN##m##LMUL##_m(_mask, _xi, _xi, 1, vl), vl);         \
    }

_RVV_FLOAT_CEIL_OP(8, 4, 32)
_RVV_FLOAT_CEIL_OP(4, 8, 32)
_RVV_FLOAT_CEIL_OP(2, 16, 32)
_RVV_FLOAT_CEIL_OP(1, 32, 32)
_RVV_FLOAT_CEIL_OP(8, 2, 16)
_RVV_FLOAT_CEIL_OP(4, 4, 16)
_RVV_FLOAT_CEIL_OP(2, 8, 16)
_RVV_FLOAT_CEIL_OP(1, 16, 16)

#define _RVV_FLOAT_FLOOR_OP(LMUL, MLEN, TLEN)                                  \
    static inline vfloat##TLEN##m##LMUL##_t floor_op(                          \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        vint##TLEN##m##LMUL##_t _xi = vfcvt_x_f_v_i##TLEN##m##LMUL(x, vl);     \
        auto _mask = vmfgt_vv_f##TLEN##m##LMUL##_b##MLEN(                      \
            vfcvt_f_x_v_f##TLEN##m##LMUL(_xi, vl), x, vl);                     \
        return vfcvt_f_x_v_f##TLEN##m##LMUL(                                   \
            vsub_vx_i##TLEN##m##LMUL##_m(_mask, _xi, _xi, 1, vl), vl);         \
    }

_RVV_FLOAT_FLOOR_OP(8, 4, 32)
_RVV_FLOAT_FLOOR_OP(4, 8, 32)
_RVV_FLOAT_FLOOR_OP(2, 16, 32)
_RVV_FLOAT_FLOOR_OP(1, 32, 32)
_RVV_FLOAT_FLOOR_OP(8, 2, 16)
_RVV_FLOAT_FLOOR_OP(4, 4, 16)
_RVV_FLOAT_FLOOR_OP(2, 8, 16)
_RVV_FLOAT_FLOOR_OP(1, 16, 16)

#define _RVV_FLOAT_ROUND_OP(LMUL, TLEN)                                        \
    static inline vfloat##TLEN##m##LMUL##_t round_op(                          \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfcvt_f_x_v_f##TLEN##m##LMUL(                                   \
            vfcvt_x_f_v_i##TLEN##m##LMUL(x, vl), vl);                          \
    }

_RVV_FLOAT_ROUND_OP(8, 32)
_RVV_FLOAT_ROUND_OP(4, 32)
_RVV_FLOAT_ROUND_OP(2, 32)
_RVV_FLOAT_ROUND_OP(1, 32)
_RVV_FLOAT_ROUND_OP(8, 16)
_RVV_FLOAT_ROUND_OP(4, 16)
_RVV_FLOAT_ROUND_OP(2, 16)
_RVV_FLOAT_ROUND_OP(1, 16)

#define _RVV_FLOAT_SQUARE_OP(LMUL, TLEN)                                       \
    static inline vfloat##TLEN##m##LMUL##_t square_op(                         \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfmul_vv_f##TLEN##m##LMUL(x, x, vl);                            \
    }

_RVV_FLOAT_SQUARE_OP(8, 32)
_RVV_FLOAT_SQUARE_OP(4, 32)
_RVV_FLOAT_SQUARE_OP(2, 32)
_RVV_FLOAT_SQUARE_OP(1, 32)
_RVV_FLOAT_SQUARE_OP(8, 16)
_RVV_FLOAT_SQUARE_OP(4, 16)
_RVV_FLOAT_SQUARE_OP(2, 16)
_RVV_FLOAT_SQUARE_OP(1, 16)

#define _RVV_FLOAT_NEG_OP(LMUL, TLEN)                                          \
    static inline vfloat##TLEN##m##LMUL##_t neg_op(                            \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfneg_v_f##TLEN##m##LMUL(x, vl);                                \
    }

_RVV_FLOAT_NEG_OP(8, 32)
_RVV_FLOAT_NEG_OP(4, 32)
_RVV_FLOAT_NEG_OP(2, 32)
_RVV_FLOAT_NEG_OP(1, 32)
_RVV_FLOAT_NEG_OP(8, 16)
_RVV_FLOAT_NEG_OP(4, 16)
_RVV_FLOAT_NEG_OP(2, 16)
_RVV_FLOAT_NEG_OP(1, 16)

#define _RVV_FLOAT_SQRT_OP(LMUL, TLEN)                                         \
    static inline vfloat##TLEN##m##LMUL##_t sqrt_op(                           \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfsqrt_v_f##TLEN##m##LMUL(x, vl);                               \
    }

_RVV_FLOAT_SQRT_OP(8, 32)
_RVV_FLOAT_SQRT_OP(4, 32)
_RVV_FLOAT_SQRT_OP(2, 32)
_RVV_FLOAT_SQRT_OP(1, 32)
_RVV_FLOAT_SQRT_OP(8, 16)
_RVV_FLOAT_SQRT_OP(4, 16)
_RVV_FLOAT_SQRT_OP(2, 16)
_RVV_FLOAT_SQRT_OP(1, 16)

#define _RVV_FLOAT_RSQRT_OP(LMUL, TLEN)                                        \
    static inline vfloat##TLEN##m##LMUL##_t rsqrt_op(                          \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        return vfrsqrt7_v_f##TLEN##m##LMUL(x, vl);                             \
    }

_RVV_FLOAT_RSQRT_OP(8, 32)
_RVV_FLOAT_RSQRT_OP(4, 32)
_RVV_FLOAT_RSQRT_OP(2, 32)
_RVV_FLOAT_RSQRT_OP(1, 32)
_RVV_FLOAT_RSQRT_OP(8, 16)
_RVV_FLOAT_RSQRT_OP(4, 16)
_RVV_FLOAT_RSQRT_OP(2, 16)
_RVV_FLOAT_RSQRT_OP(1, 16)

#define _RVV_FLOAT_SWISH_OP(LMUL, TLEN)                                        \
    static inline vfloat##TLEN##m##LMUL##_t swish_op(                          \
        vfloat##TLEN##m##LMUL##_t x, size_t vl, float##TLEN##_t beta) {        \
        auto vtmp = vf_mul_op(x, beta, vl);                                    \
        vtmp = neg_op(vtmp, vl);                                               \
        vtmp = exp_ps(vtmp, vl);                                               \
        vtmp = vf_add_op(vtmp, (float##TLEN##_t)1.0f, vl);                     \
        return vv_div_op(x, vtmp, vl);                                         \
    }

_RVV_FLOAT_SWISH_OP(8, 32)
_RVV_FLOAT_SWISH_OP(4, 32)
_RVV_FLOAT_SWISH_OP(2, 32)
_RVV_FLOAT_SWISH_OP(1, 32)
_RVV_FLOAT_SWISH_OP(8, 16)
_RVV_FLOAT_SWISH_OP(4, 16)
_RVV_FLOAT_SWISH_OP(2, 16)
_RVV_FLOAT_SWISH_OP(1, 16)

#endif