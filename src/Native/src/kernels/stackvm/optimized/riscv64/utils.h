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

#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        x = __riscv_vfmax_vf_f32m##LMUL(                                       \
            x, 0.f, vl); /* force flush to zero on denormal values */          \
        vbool##MLEN##_t invalid_mask =                                         \
            __riscv_vmfle_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                 \
                                                                               \
        vint32m##LMUL##_t ux =                                                 \
            __riscv_vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);                 \
                                                                               \
        vint32m##LMUL##_t emm0 = __riscv_vsra_vx_i32m##LMUL(ux, 23, vl);       \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = __riscv_vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);              \
        ux = __riscv_vor_vx_i32m##LMUL(                                        \
            ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);              \
        x = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                \
                                                                               \
        emm0 = __riscv_vsub_vx_i32m##LMUL(emm0, 0x7f, vl);                     \
        vfloat32m##LMUL##_t e = __riscv_vfcvt_f_x_v_f32m##LMUL(emm0, vl);      \
                                                                               \
        e = __riscv_vfadd_vf_f32m##LMUL(e, 1.f, vl);                           \
                                                                               \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##MLEN##_t mask =                                                 \
            __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);     \
        x = __riscv_vfadd_vv_f32m##LMUL##_m(mask, x, x, vl);                   \
        x = __riscv_vfsub_vf_f32m##LMUL(x, 1.f, vl);                           \
        e = __riscv_vfsub_vf_f32m##LMUL##_m(mask, e, 1.f, vl);                 \
                                                                               \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);         \
                                                                               \
        vfloat32m##LMUL##_t y =                                                \
            __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_log_p0, vl);               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p1, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p2, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p3, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p4, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p5, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p6, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p7, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p8, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
                                                                               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
                                                                               \
        vfloat32m##LMUL##_t tmp =                                              \
            __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q1, vl);               \
        y = __riscv_vfadd_vv_f32m##LMUL(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f32m##LMUL(z, 0.5f, vl);                        \
        y = __riscv_vfsub_vv_f32m##LMUL(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q2, vl);             \
        x = __riscv_vfadd_vv_f32m##LMUL(x, y, vl);                             \
        x = __riscv_vfadd_vv_f32m##LMUL(x, tmp, vl);                           \
        /* negative arg will be NAN */                                         \
        vuint32m##LMUL##_t xtmp =                                              \
            __riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(x);                 \
        x = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(                    \
            __riscv_vor_vx_u32m##LMUL##_m(invalid_mask, xtmp, 0xffffffff,      \
                                          vl));                                \
        return x;                                                              \
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

#if 0
#define _RVV_FLOAT32_EXP_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t exp_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        vfloat32m##LMUL##_t tmp, fx;                                           \
                                                                               \
        x = __riscv_vfmin_vf_f32m##LMUL(x, c_exp_hi, vl);                      \
        x = __riscv_vfmax_vf_f32m##LMUL(x, c_exp_lo, vl);                      \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        fx = __riscv_vfmacc_vf_f32m##LMUL(                                     \
            __riscv_vfmv_v_f_f32m##LMUL(0.5f, vl), c_cephes_LOG2EF, x, vl);    \
                                                                               \
        /* perform a floorf */                                                 \
        tmp = __riscv_vfcvt_f_x_v_f32m##LMUL(                                  \
            __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl), vl);                       \
                                                                               \
        /* if greater, substract 1 */                                          \
        vbool##MLEN##_t mask =                                                 \
            __riscv_vmfgt_vv_f32m##LMUL##_b##MLEN(tmp, fx, vl);                \
        fx = __riscv_vfsub_vf_f32m##LMUL##_m(mask, tmp, 1.f, vl);              \
                                                                               \
        tmp = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C1, vl);            \
        vfloat32m##LMUL##_t z =                                                \
            __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C2, vl);              \
        x = __riscv_vfsub_vv_f32m##LMUL(x, tmp, vl);                           \
        x = __riscv_vfsub_vv_f32m##LMUL(x, z, vl);                             \
                                                                               \
        vfloat32m##LMUL##_t y =                                                \
            __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_exp_p0, vl);               \
        z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                             \
                                                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p1, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p2, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p3, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p4, vl);               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p5, vl);               \
                                                                               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, 1.f, vl);                           \
                                                                               \
        /* build 2^n */                                                        \
        vint32m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl);         \
        mm = __riscv_vadd_vx_i32m##LMUL(mm, 0x7f, vl);                         \
        mm = __riscv_vsll_vx_i32m##LMUL(mm, 23, vl);                           \
        vfloat32m##LMUL##_t pow2n =                                            \
            __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(mm);                \
                                                                               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, pow2n, vl);                         \
        return y;                                                              \
    }

_RVV_FLOAT32_EXP_OP(1, 32)
_RVV_FLOAT32_EXP_OP(2, 16)
_RVV_FLOAT32_EXP_OP(4, 8)
_RVV_FLOAT32_EXP_OP(8, 4)
#else
// e^x = 1 + x + 1/2!x^2 + 1/3!x^3 + 1/4!x^4 + 1/5!x^5 + 1/6!x^6 + 1/7!x^7
#define _RVV_FLOAT32_EXP_OP(LMUL, MLEN, TLEN, E, M)                            \
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

_RVV_FLOAT32_EXP_OP(1, 32, 32, 0x7f, 23)
_RVV_FLOAT32_EXP_OP(2, 16, 32, 0x7f, 23)
_RVV_FLOAT32_EXP_OP(4, 8, 32, 0x7f, 23)
_RVV_FLOAT32_EXP_OP(8, 4, 32, 0x7f, 23)
#endif

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

#define _RVV_FLOAT32_SINCOS_OP(LMUL, MLEN)                                     \
    static inline void sincos_ps(vfloat32m##LMUL##_t x,                        \
                                 vfloat32m##LMUL##_t *ysin,                    \
                                 vfloat32m##LMUL##_t *ycos, size_t vl) {       \
        /* any x */                                                            \
        vfloat32m##LMUL##_t xmm1, xmm2, xmm3, y;                               \
                                                                               \
        vuint32m##LMUL##_t emm2;                                               \
                                                                               \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                          \
        sign_mask_sin = __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);     \
        x = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.f, vl);                          \
                                                                               \
        /* scale by 4/Pi */                                                    \
        y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_FOPI, vl);                 \
                                                                               \
        /* store the integer part of y in mm0 */                               \
        emm2 = __riscv_vfcvt_xu_f_v_u32m##LMUL(y, vl);                         \
        /* j=(j+1) & (~1) (see the cephes sources) */                          \
        emm2 = __riscv_vadd_vx_u32m##LMUL(emm2, 1, vl);                        \
        emm2 = __riscv_vand_vx_u32m##LMUL(emm2, ~1, vl);                       \
        y = __riscv_vfcvt_f_xu_v_f32m##LMUL(emm2, vl);                         \
                                                                               \
        /* get the polynom selection mask              */                      \
        /*     there is one polynom for 0 <= x <= Pi/4 */                      \
        /*     and another one for Pi/4<x<=Pi/2        */                      \
        /*                                             */                      \
        /*     Both branches will be computed.         */                      \
        vbool##MLEN##_t poly_mask = __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(     \
            __riscv_vand_vx_u32m##LMUL(emm2, 2, vl), 0, vl);                   \
                                                                               \
        /* The magic pass: "Extended precision modular arithmetic" */          \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */          \
        xmm1 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP1, vl);         \
        xmm2 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP2, vl);         \
        xmm3 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP3, vl);         \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm1, vl);                          \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm2, vl);                          \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm3, vl);                          \
                                                                               \
        sign_mask_sin = __riscv_vmxor_mm_b##MLEN(                              \
            sign_mask_sin,                                                     \
            __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(                             \
                __riscv_vand_vx_u32m##LMUL(emm2, 4, vl), 0, vl),               \
            vl);                                                               \
        sign_mask_cos = __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(                 \
            __riscv_vand_vx_u32m##LMUL(                                        \
                __riscv_vsub_vx_u32m##LMUL(emm2, 2, vl), 4, vl),               \
            0, vl);                                                            \
                                                                               \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */              \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */              \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);         \
        vfloat32m##LMUL##_t y1, y2;                                            \
                                                                               \
        y1 = __riscv_vfmul_vf_f32m##LMUL(z, c_coscof_p0, vl);                  \
        y2 = __riscv_vfmul_vf_f32m##LMUL(z, c_sincof_p0, vl);                  \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, c_coscof_p1, vl);                 \
        y2 = __riscv_vfadd_vf_f32m##LMUL(y2, c_sincof_p1, vl);                 \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, c_coscof_p2, vl);                 \
        y2 = __riscv_vfadd_vf_f32m##LMUL(y2, c_sincof_p2, vl);                 \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfsub_vv_f32m##LMUL(                                      \
            y1, __riscv_vfmul_vf_f32m##LMUL(z, 0.5f, vl), vl);                 \
        y2 = __riscv_vfadd_vv_f32m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, 1.f, vl);                         \
                                                                               \
        /* select the correct result from the two polynoms */                  \
        vfloat32m##LMUL##_t ys =                                               \
            __riscv_vmerge_vvm_f32m##LMUL(y2, y1, poly_mask, vl);              \
        vfloat32m##LMUL##_t yc =                                               \
            __riscv_vmerge_vvm_f32m##LMUL(y1, y2, poly_mask, vl);              \
        *ysin = __riscv_vmerge_vvm_f32m##LMUL(                                 \
            ys, __riscv_vfneg_v_f32m##LMUL(ys, vl), sign_mask_sin, vl);        \
        *ycos = __riscv_vmerge_vvm_f32m##LMUL(                                 \
            __riscv_vfneg_v_f32m##LMUL(yc, vl), yc, sign_mask_cos, vl);        \
    }

_RVV_FLOAT32_SINCOS_OP(1, 32)
_RVV_FLOAT32_SINCOS_OP(2, 16)
_RVV_FLOAT32_SINCOS_OP(4, 8)
_RVV_FLOAT32_SINCOS_OP(8, 4)

#define _RVV_FLOAT32_SIN_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t sin_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        vfloat32m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ysin;                                                           \
    }

_RVV_FLOAT32_SIN_OP(1, 32)
_RVV_FLOAT32_SIN_OP(2, 16)
_RVV_FLOAT32_SIN_OP(4, 8)
_RVV_FLOAT32_SIN_OP(8, 4)

#define _RVV_FLOAT32_COS_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t cos_ps(vfloat32m##LMUL##_t x,            \
                                             size_t vl) {                      \
        vfloat32m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ycos;                                                           \
    }

_RVV_FLOAT32_COS_OP(1, 32)
_RVV_FLOAT32_COS_OP(2, 16)
_RVV_FLOAT32_COS_OP(4, 8)
_RVV_FLOAT32_COS_OP(8, 4)

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1 0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

#define _RVV_FLOAT32_TANH_OP(LMUL, MLEN)                                       \
    static inline vfloat32m##LMUL##_t tanh_ps(vfloat32m##LMUL##_t x,           \
                                              size_t vl) {                     \
        vfloat32m##LMUL##_t x2 = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.f, vl);     \
                                                                               \
        vbool##MLEN##_t mask_l =                                               \
            __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x2, c_cephes_tanh_C1, vl);   \
        vbool##MLEN##_t mask_l2 = __riscv_vmfgt_vf_f32m##LMUL##_b##MLEN(       \
            x2, c_cephes_HALFMAXLOGF, vl);                                     \
                                                                               \
        /* abs(x) >= 0.625 */                                                  \
        vfloat32m##LMUL##_t exp_x_x =                                          \
            exp_ps(__riscv_vfadd_vv_f32m##LMUL(x, x, vl), vl);                 \
        vfloat32m##LMUL##_t y0 = __riscv_vfrsub_vf_f32m##LMUL(                 \
            __riscv_vfrdiv_vf_f32m##LMUL(                                      \
                __riscv_vfadd_vf_f32m##LMUL(exp_x_x, 1.f, vl), 2.f, vl),       \
            1.f, vl);                                                          \
                                                                               \
        /* abs(x) < 0.625                */                                    \
        /*   z = x2 * x2;                */                                    \
        /*   z =                         */                                    \
        /*   (((( -5.70498872745E-3 * z  */                                    \
        /*   + 2.06390887954E-2) * z     */                                    \
        /*   - 5.37397155531E-2) * z     */                                    \
        /*   + 1.33314422036E-1) * z     */                                    \
        /*   - 3.33332819422E-1) * z * x */                                    \
        /*   + x;                        */                                    \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);         \
                                                                               \
        vfloat32m##LMUL##_t y =                                                \
            __riscv_vfmul_vf_f32m##LMUL(z, c_cephes_tanh_p0, vl);              \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_tanh_p1, vl);              \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_tanh_p2, vl);              \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_tanh_p3, vl);              \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_tanh_p4, vl);              \
                                                                               \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                             \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                             \
                                                                               \
        /* abs(x) > HALFMAXLOGF */                                             \
        vfloat32m##LMUL##_t y1 = __riscv_vfsgnj_vv_f32m##LMUL(                 \
            __riscv_vfmv_v_f_f32m##LMUL(1.f, vl), x, vl);                      \
                                                                               \
        y = __riscv_vmerge_vvm_f32m##LMUL(y, y0, mask_l, vl);                  \
        y = __riscv_vmerge_vvm_f32m##LMUL(y, y1, mask_l2, vl);                 \
        return y;                                                              \
    }

_RVV_FLOAT32_TANH_OP(1, 32)
_RVV_FLOAT32_TANH_OP(2, 16)
_RVV_FLOAT32_TANH_OP(4, 8)
_RVV_FLOAT32_TANH_OP(8, 4)

#define _RVV_FLOAT32_POW_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t pow_ps(                                  \
        vfloat32m##LMUL##_t a, vfloat32m##LMUL##_t b, size_t vl) {             \
        /* pow(x, m) = exp(m * log(x)) */                                      \
        return exp_ps(__riscv_vfmul_vv_f32m##LMUL(b, log_ps(a, vl), vl), vl);  \
    }

_RVV_FLOAT32_POW_OP(1, 32)
_RVV_FLOAT32_POW_OP(2, 16)
_RVV_FLOAT32_POW_OP(4, 8)
_RVV_FLOAT32_POW_OP(8, 4)

#endif