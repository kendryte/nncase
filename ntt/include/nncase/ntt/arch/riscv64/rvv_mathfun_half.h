#pragma once
#include "nncase/half.h"
#include "rvv_mathfun.h"
#include <cmath>

#if __riscv_vector
#include <riscv_vector.h>

using namespace nncase;

// log
#define c_inv_mant_mask_f16 -31745
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

#define _RVV_FLOAT16_LOG_OP(lmul, mlen)                                        \
    inline vfloat16m##lmul##_t log_ps_fp16(vfloat16m##lmul##_t x, size_t vl) { \
        x = __riscv_vfmax_vf_f16m##lmul(                                       \
            x, half::round_to_half(0.f),                                       \
            vl); /* force flush to zero on denormal values */                  \
        vbool##mlen##_t invalid_mask = __riscv_vmfle_vf_f16m##lmul##_b##mlen(  \
            x, half::round_to_half(0.f), vl);                                  \
                                                                               \
        vint16m##lmul##_t ux =                                                 \
            __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(x);                 \
                                                                               \
        vint16m##lmul##_t emm0 = __riscv_vsra_vx_i16m##lmul(ux, 10, vl);       \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = __riscv_vand_vx_i16m##lmul(ux, c_inv_mant_mask_f16, vl);          \
        ux = __riscv_vor_vx_i16m##lmul(                                        \
            ux, 14336 /* reinterpret_cast<short>((_Float16)0.5) */, vl);       \
        x = __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(ux);                \
                                                                               \
        emm0 = __riscv_vsub_vx_i16m##lmul(emm0, 0xf, vl);                      \
        vfloat16m##lmul##_t e = __riscv_vfcvt_f_x_v_f16m##lmul(emm0, vl);      \
                                                                               \
        e = __riscv_vfadd_vf_f16m##lmul(e, half::round_to_half(1.f), vl);      \
                                                                               \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##mlen##_t mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(          \
            x, half::round_to_half(c_cephes_SQRTHF), vl);                      \
        x = __riscv_vfadd_vv_f16m##lmul##_mu(mask, x, x, x, vl);               \
        x = __riscv_vfsub_vf_f16m##lmul(x, half::round_to_half(1.f), vl);      \
        e = __riscv_vfsub_vf_f16m##lmul##_mu(mask, e, e,                       \
                                             half::round_to_half(1.f), vl);    \
                                                                               \
        vfloat16m##lmul##_t z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);         \
                                                                               \
        vfloat16m##lmul##_t y = __riscv_vfmul_vf_f16m##lmul(                   \
            x, half::round_to_half(c_cephes_log_p0), vl);                      \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p1), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p2), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p3), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p4), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p5), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p6), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p7), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_log_p8), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
                                                                               \
        y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                             \
                                                                               \
        vfloat16m##lmul##_t tmp = __riscv_vfmul_vf_f16m##lmul(                 \
            e, half::round_to_half(c_cephes_log_q1), vl);                      \
        y = __riscv_vfadd_vv_f16m##lmul(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(z, half::round_to_half(0.5f), vl);   \
        y = __riscv_vfsub_vv_f16m##lmul(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(                                     \
            e, half::round_to_half(c_cephes_log_q2), vl);                      \
        x = __riscv_vfadd_vv_f16m##lmul(x, y, vl);                             \
        x = __riscv_vfadd_vv_f16m##lmul(x, tmp, vl);                           \
        /* negative arg will be NAN */                                         \
        vuint16m##lmul##_t xtmp =                                              \
            __riscv_vreinterpret_v_f16m##lmul##_u16m##lmul(x);                 \
        x = __riscv_vreinterpret_v_u16m##lmul##_f16m##lmul(                    \
            __riscv_vor_vx_u16m##lmul##_mu(invalid_mask, xtmp, xtmp, 0xffff,   \
                                           vl));                               \
        return x;                                                              \
    }

_RVV_FLOAT16_LOG_OP(1, 16)
_RVV_FLOAT16_LOG_OP(2, 8)
_RVV_FLOAT16_LOG_OP(4, 4)
_RVV_FLOAT16_LOG_OP(8, 2)

// exp
#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#define _RVV_FLOAT16_EXP_OP(lmul, mlen)                                        \
    static inline vfloat16m##lmul##_t exp_ps_fp16(vfloat16m##lmul##_t x,       \
                                                  size_t vl) {                 \
        vfloat16m##lmul##_t tmp, fx;                                           \
                                                                               \
        x = __riscv_vfmin_vf_f16m##lmul(x, half::round_to_half(c_exp_hi_f16),  \
                                        vl);                                   \
        x = __riscv_vfmax_vf_f16m##lmul(x, half::round_to_half(c_exp_lo_f16),  \
                                        vl);                                   \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        fx = __riscv_vfmacc_vf_f16m##lmul(                                     \
            __riscv_vfmv_v_f_f16m##lmul(half::round_to_half(0.5f), vl),        \
            half::round_to_half(c_cephes_LOG2EF), x, vl);                      \
                                                                               \
        /* perform a floorf */                                                 \
        tmp = __riscv_vfcvt_f_x_v_f16m##lmul(                                  \
            __riscv_vfcvt_x_f_v_i16m##lmul(fx, vl), vl);                       \
                                                                               \
        /* if greater, substract 1 */                                          \
        vbool##mlen##_t mask =                                                 \
            __riscv_vmfgt_vv_f16m##lmul##_b##mlen(tmp, fx, vl);                \
        fx = __riscv_vfsub_vf_f16m##lmul##_mu(mask, tmp, tmp,                  \
                                              half::round_to_half(1.f), vl);   \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(                                     \
            fx, half::round_to_half(c_cephes_exp_C1), vl);                     \
        vfloat16m##lmul##_t z = __riscv_vfmul_vf_f16m##lmul(                   \
            fx, half::round_to_half(c_cephes_exp_C2), vl);                     \
        x = __riscv_vfsub_vv_f16m##lmul(x, tmp, vl);                           \
        x = __riscv_vfsub_vv_f16m##lmul(x, z, vl);                             \
                                                                               \
        vfloat16m##lmul##_t y = __riscv_vfmul_vf_f16m##lmul(                   \
            x, half::round_to_half(c_cephes_exp_p0), vl);                      \
        z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);                             \
                                                                               \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_exp_p1), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_exp_p2), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_exp_p3), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_exp_p4), vl);                      \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(                                       \
            y, half::round_to_half(c_cephes_exp_p5), vl);                      \
                                                                               \
        y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                             \
        y = __riscv_vfadd_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, half::round_to_half(1.f), vl);      \
                                                                               \
        /* build 2^n */                                                        \
        vint16m##lmul##_t mm = __riscv_vfcvt_x_f_v_i16m##lmul(fx, vl);         \
        mm = __riscv_vadd_vx_i16m##lmul(mm, 0xf, vl);                          \
        mm = __riscv_vsll_vx_i16m##lmul(mm, 10, vl);                           \
        vfloat16m##lmul##_t pow2n =                                            \
            __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(mm);                \
                                                                               \
        y = __riscv_vfmul_vv_f16m##lmul(y, pow2n, vl);                         \
        return y;                                                              \
    }

_RVV_FLOAT16_EXP_OP(1, 16)
_RVV_FLOAT16_EXP_OP(2, 8)
_RVV_FLOAT16_EXP_OP(4, 4)
_RVV_FLOAT16_EXP_OP(8, 2)

// sincos
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

#define _RVV_FLOAT16_SINCOS_OP(LMUL, MLEN)                                     \
    static inline void sincos_ps_fp16(vfloat16m##LMUL##_t x,                   \
                                      vfloat16m##LMUL##_t *ysin,               \
                                      vfloat16m##LMUL##_t *ycos, size_t vl) {  \
        /* any x */                                                            \
        vfloat16m##LMUL##_t xmm1, xmm2, xmm3, y;                               \
                                                                               \
        vuint16m##LMUL##_t emm2;                                               \
                                                                               \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                          \
        sign_mask_sin =                                                        \
            __riscv_vmflt_vf_f16m##LMUL##_b##MLEN(x, (_Float16)0.f, vl);       \
        x = __riscv_vfsgnj_vf_f16m##LMUL(x, (_Float16)1.f, vl);                \
                                                                               \
        /* scale by 4/Pi */                                                    \
        y = __riscv_vfmul_vf_f16m##LMUL(x, (_Float16)c_cephes_FOPI, vl);       \
                                                                               \
        /* store the integer part of y in mm0 */                               \
        emm2 = __riscv_vfcvt_xu_f_v_u16m##LMUL(y, vl);                         \
        /* j=(j+1) & (~1) (see the cephes sources) */                          \
        emm2 = __riscv_vadd_vx_u16m##LMUL(emm2, 1, vl);                        \
        emm2 = __riscv_vand_vx_u16m##LMUL(emm2, ~1, vl);                       \
        y = __riscv_vfcvt_f_xu_v_f16m##LMUL(emm2, vl);                         \
                                                                               \
        /* get the polynom selection mask              */                      \
        /*     there is one polynom for 0 <= x <= Pi/4 */                      \
        /*     and another one for Pi/4<x<=Pi/2        */                      \
        /*                                             */                      \
        /*     Both branches will be computed.         */                      \
        vbool##MLEN##_t poly_mask = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(     \
            __riscv_vand_vx_u16m##LMUL(emm2, 2, vl), 0, vl);                   \
                                                                               \
        /* The magic pass: "Extended precision modular arithmetic" */          \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */          \
        xmm1 =                                                                 \
            __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP1, vl);  \
        xmm2 =                                                                 \
            __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP2, vl);  \
        xmm3 =                                                                 \
            __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP3, vl);  \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm1, vl);                          \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm2, vl);                          \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm3, vl);                          \
                                                                               \
        sign_mask_sin = __riscv_vmxor_mm_b##MLEN(                              \
            sign_mask_sin,                                                     \
            __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(                             \
                __riscv_vand_vx_u16m##LMUL(emm2, 4, vl), 0, vl),               \
            vl);                                                               \
        sign_mask_cos = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(                 \
            __riscv_vand_vx_u16m##LMUL(                                        \
                __riscv_vsub_vx_u16m##LMUL(emm2, 2, vl), 4, vl),               \
            0, vl);                                                            \
                                                                               \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */              \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */              \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vv_f16m##LMUL(x, x, vl);         \
        vfloat16m##LMUL##_t y1, y2;                                            \
                                                                               \
        y1 = __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)c_coscof_p0, vl);        \
        y2 = __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)c_sincof_p0, vl);        \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)c_coscof_p1, vl);       \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (_Float16)c_sincof_p1, vl);       \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)c_coscof_p2, vl);       \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (_Float16)c_sincof_p2, vl);       \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfsub_vv_f16m##LMUL(                                      \
            y1, __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)0.5f, vl), vl);       \
        y2 = __riscv_vfadd_vv_f16m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)1.f, vl);               \
                                                                               \
        /* select the correct result from the two polynoms */                  \
        vfloat16m##LMUL##_t ys =                                               \
            __riscv_vmerge_vvm_f16m##LMUL(y2, y1, poly_mask, vl);              \
        vfloat16m##LMUL##_t yc =                                               \
            __riscv_vmerge_vvm_f16m##LMUL(y1, y2, poly_mask, vl);              \
        *ysin = __riscv_vmerge_vvm_f16m##LMUL(                                 \
            ys, __riscv_vfneg_v_f16m##LMUL(ys, vl), sign_mask_sin, vl);        \
        *ycos = __riscv_vmerge_vvm_f16m##LMUL(                                 \
            __riscv_vfneg_v_f16m##LMUL(yc, vl), yc, sign_mask_cos, vl);        \
    }

_RVV_FLOAT16_SINCOS_OP(1, 16)
_RVV_FLOAT16_SINCOS_OP(2, 8)
_RVV_FLOAT16_SINCOS_OP(4, 4)
_RVV_FLOAT16_SINCOS_OP(8, 2)

// tanh
#define LOG2_INV 0x1.71547652b82fep+0
#define LOG2_HI 0x1.62e42fefa39efp-1
#define LOG2_LO 0x1.abc9e3b39803fp-56

#define _RVV_FLOAT16_TANH_OP(LMUL, MLEN, TLEN)                                 \
    static inline vfloat##TLEN##m##LMUL##_t tanh_ps_fp16(                      \
        vfloat##TLEN##m##LMUL##_t v, size_t vl) {                              \
        constexpr auto fp_posZero = 0.f16;                                     \
        constexpr auto fp_posOne = 1.f16;                                      \
        auto zero = __riscv_vfmv_v_f_f##TLEN##m##LMUL(fp_posZero, vl);         \
        auto one = __riscv_vfmv_v_f_f##TLEN##m##LMUL(fp_posOne, vl);           \
        /*tanh(x) = sign(x) * tanh(|x|); suffices to work on |x| for the main  \
         * part */                                                             \
        auto vx = __riscv_vfsgnj_vf_f##TLEN##m##LMUL(v, fp_posOne, vl);        \
        /* Suffices to clip |x| to 20, which is bigger than 28 log(2) */       \
        vx = __riscv_vfmin_vf_f##TLEN##m##LMUL(                                \
            vx, half::round_to_half(0x1.4p4), vl);                             \
                                                                               \
        /* tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)); so we compute exp(-2x)   \
         */                                                                    \
        /* by replacing x by -2x */                                            \
        vx = __riscv_vfmul_vf_f##TLEN##m##LMUL(vx, half::round_to_half(-2.f),  \
                                               vl);                            \
        auto n_flt = __riscv_vfmul_vf_f##TLEN##m##LMUL(                        \
            vx, half::round_to_half(LOG2_INV), vl);                            \
        auto n = __riscv_vfcvt_x_f_v_i##TLEN##m##LMUL(n_flt, vl);              \
        n_flt = __riscv_vfcvt_f_x_v_f##TLEN##m##LMUL(n, vl);                   \
        auto u = __riscv_vadd_vx_i##TLEN##m##LMUL(n, 127, vl);                 \
        auto r_delta = __riscv_vfnmsac_vf_f##TLEN##m##LMUL(                    \
            vx, half::round_to_half(LOG2_HI), n_flt, vl);                      \
        u = __riscv_vsll_vx_i##TLEN##m##LMUL(u, 23, vl);                       \
        auto r = __riscv_vfnmsac_vf_f##TLEN##m##LMUL(                          \
            r_delta, half::round_to_half(LOG2_LO), n_flt, vl);                 \
        auto s =                                                               \
            __riscv_vreinterpret_v_i##TLEN##m##LMUL##_f##TLEN##m##LMUL(u);     \
        auto s_is_small =                                                      \
            __riscv_vmsle_vx_i##TLEN##m##LMUL##_b##MLEN(n, -(23 + 1), vl);     \
        r_delta = __riscv_vfsub_vv_f##TLEN##m##LMUL(r_delta, r, vl);           \
        auto s_head = __riscv_vfmerge_vfm_f##TLEN##m##LMUL(s, fp_posZero,      \
                                                           s_is_small, vl);    \
        r_delta = __riscv_vfnmsac_vf_f##TLEN##m##LMUL(                         \
            r_delta, half::round_to_half(LOG2_LO), n_flt, vl);                 \
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
        auto c0 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.71ddef82f4beep-19), vl);                   \
        auto c1 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.a01a01b32b633p-13), vl);                   \
        auto c2 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.111111110ef6ap-7), vl);                    \
        auto c3 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.555555555555ap-3), vl);                    \
        auto c4 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.a019b37a2b3dfp-16), vl);                   \
        auto c5 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.6c16c17a09506p-10), vl);                   \
        auto c6 = __riscv_vfmv_v_f_f##TLEN##m##LMUL(                           \
            half::round_to_half(0x1.5555555553aefp-5), vl);                    \
                                                                               \
        auto p_even = __riscv_vmv_v_v_f##TLEN##m##LMUL(rsq, vl);               \
        p_even = __riscv_vfmadd_vf_f##TLEN##m##LMUL(                           \
            p_even, half::round_to_half(0x1.af6eacd796f0bp-26), c0, vl);       \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c1, vl);      \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c2, vl);      \
        p_even = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_even, rsq, c3, vl);      \
                                                                               \
        auto p_odd = __riscv_vmv_v_v_f##TLEN##m##LMUL(rsq, vl);                \
        p_odd = __riscv_vfmadd_vf_f##TLEN##m##LMUL(                            \
            p_odd, half::round_to_half(0x1.289788d8bdadfp-22), c4, vl);        \
        p_odd = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, rsq, c5, vl);        \
        p_odd = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, rsq, c6, vl);        \
        auto poly = __riscv_vfmadd_vv_f##TLEN##m##LMUL(p_odd, r, p_even, vl);  \
                                                                               \
        /* r^3 * poly will be r^3(...)                                         \
           we delay this multiplication with r^3 for now */                    \
                                                                               \
        /*  Compute r' + (r')^2/2 extra precisely */                           \
        auto r_prime = __riscv_vfmul_vf_f##TLEN##m##LMUL(                      \
            r, half::round_to_half(0x1.0p-1), vl);                             \
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
        return __riscv_vfsgnj_vv_f##TLEN##m##LMUL(vy, v, vl);                  \
    }

_RVV_FLOAT16_TANH_OP(1, 16, 16)
_RVV_FLOAT16_TANH_OP(2, 8, 16)
_RVV_FLOAT16_TANH_OP(4, 4, 16)
_RVV_FLOAT16_TANH_OP(8, 2, 16)

#define _RVV_FLOAT16_POW_OP(LMUL, MLEN, TLEN)                                  \
    static inline vfloat##TLEN##m##LMUL##_t pow_ps_fp16(                       \
        vfloat##TLEN##m##LMUL##_t a, vfloat##TLEN##m##LMUL##_t b, size_t vl) { \
        /* pow(x, m) = exp(m * log(x)) */                                      \
        return exp_ps_fp16(                                                    \
            __riscv_vfmul_vv_f##TLEN##m##LMUL(b, log_ps_fp16(a, vl), vl), vl); \
    }

_RVV_FLOAT16_POW_OP(1, 16, 16)
_RVV_FLOAT16_POW_OP(2, 8, 16)
_RVV_FLOAT16_POW_OP(4, 4, 16)
_RVV_FLOAT16_POW_OP(8, 2, 16)

// erf

#endif