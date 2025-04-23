#pragma once
#include "nncase/half.h"
#include "rvv_mathfun.h"
#include <cmath>

#if __riscv_vector
#include <riscv_vector.h>

using namespace nncase;

// log
#define _RVV_FLOAT16_LOG_OP(lmul, mlen)                                        \
    inline vfloat16m##lmul##_t log_ps_fp16(vfloat16m##lmul##_t x, size_t vl) { \
        constexpr auto c_inv_mant_mask_half = -31745;                          \
        constexpr auto c_cephes_SQRTHF_half = (_Float16)0.707106781186547524;  \
        constexpr auto c_cephes_log_p0_half = (_Float16)7.0376836292E-2;       \
        constexpr auto c_cephes_log_p1_half = (_Float16)-1.1514610310E-1;      \
        constexpr auto c_cephes_log_p2_half = (_Float16)1.1676998740E-1;       \
        constexpr auto c_cephes_log_p3_half = (_Float16)-1.2420140846E-1;      \
        constexpr auto c_cephes_log_p4_half = (_Float16)+1.4249322787E-1;      \
        constexpr auto c_cephes_log_p5_half = (_Float16)-1.6668057665E-1;      \
        constexpr auto c_cephes_log_p6_half = (_Float16)+2.0000714765E-1;      \
        constexpr auto c_cephes_log_p7_half = (_Float16)-2.4999993993E-1;      \
        constexpr auto c_cephes_log_p8_half = (_Float16)+3.3333331174E-1;      \
        constexpr auto c_cephes_log_q1_half = (_Float16)-2.12194440e-4;        \
        constexpr auto c_cephes_log_q2_half = (_Float16)0.693359375;           \
        x = __riscv_vfmax_vf_f16m##lmul(                                       \
            x, 0.f16, vl); /* force flush to zero on denormal values */        \
        vbool##mlen##_t invalid_mask =                                         \
            __riscv_vmfle_vf_f16m##lmul##_b##mlen(x, 0.f16, vl);               \
                                                                               \
        vint16m##lmul##_t ux =                                                 \
            __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(x);                 \
                                                                               \
        vint16m##lmul##_t emm0 = __riscv_vsra_vx_i16m##lmul(ux, 10, vl);       \
                                                                               \
        /* keep only the fractional part */                                    \
        ux = __riscv_vand_vx_i16m##lmul(ux, c_inv_mant_mask_half, vl);         \
        ux = __riscv_vor_vx_i16m##lmul(                                        \
            ux, 14336 /* reinterpret_cast<short>((_Float16)0.5) */, vl);       \
        x = __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(ux);                \
                                                                               \
        emm0 = __riscv_vsub_vx_i16m##lmul(emm0, 0xf, vl);                      \
        vfloat16m##lmul##_t e = __riscv_vfcvt_f_x_v_f16m##lmul(emm0, vl);      \
                                                                               \
        e = __riscv_vfadd_vf_f16m##lmul(e, 1.f16, vl);                         \
                                                                               \
        /* part2:                      */                                      \
        /*     if( x < SQRTHF ) {      */                                      \
        /*       e -= 1;               */                                      \
        /*       x = x + x - 1.0;      */                                      \
        /*     } else { x = x - 1.0; } */                                      \
        vbool##mlen##_t mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(          \
            x, c_cephes_SQRTHF_half, vl);                                      \
        x = __riscv_vfadd_vv_f16m##lmul##_mu(mask, x, x, x, vl);               \
        x = __riscv_vfsub_vf_f16m##lmul(x, 1.f16, vl);                         \
        e = __riscv_vfsub_vf_f16m##lmul##_mu(mask, e, e, 1.f16, vl);           \
                                                                               \
        vfloat16m##lmul##_t z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);         \
                                                                               \
        vfloat16m##lmul##_t y =                                                \
            __riscv_vfmul_vf_f16m##lmul(x, c_cephes_log_p0_half, vl);          \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p1_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p2_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p3_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p4_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p5_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p6_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p7_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_log_p8_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
                                                                               \
        y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                             \
                                                                               \
        vfloat16m##lmul##_t tmp =                                              \
            __riscv_vfmul_vf_f16m##lmul(e, c_cephes_log_q1_half, vl);          \
        y = __riscv_vfadd_vv_f16m##lmul(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(z, 0.5f16, vl);                      \
        y = __riscv_vfsub_vv_f16m##lmul(y, tmp, vl);                           \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(e, c_cephes_log_q2_half, vl);        \
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
#define _RVV_FLOAT16_EXP_OP(lmul, mlen)                                        \
    static inline vfloat16m##lmul##_t exp_ps_fp16(vfloat16m##lmul##_t x,       \
                                                  size_t vl) {                 \
        constexpr auto c_exp_hi_half = (_Float16)10.7421875f;                  \
        constexpr auto c_exp_lo_half = (_Float16)-10.7421875f;                 \
        constexpr auto c_cephes_LOG2EF_half = (_Float16)1.44269504088896341;   \
        constexpr auto c_cephes_exp_C1_half = (_Float16)0.693359375;           \
        constexpr auto c_cephes_exp_C2_half = (_Float16)-2.12194440e-4;        \
        constexpr auto c_cephes_exp_p0_half = (_Float16)1.9875691500E-4;       \
        constexpr auto c_cephes_exp_p1_half = (_Float16)1.3981999507E-3;       \
        constexpr auto c_cephes_exp_p2_half = (_Float16)8.3334519073E-3;       \
        constexpr auto c_cephes_exp_p3_half = (_Float16)4.1665795894E-2;       \
        constexpr auto c_cephes_exp_p4_half = (_Float16)1.6666665459E-1;       \
        constexpr auto c_cephes_exp_p5_half = (_Float16)5.0000001201E-1;       \
        vfloat16m##lmul##_t tmp, fx;                                           \
                                                                               \
        x = __riscv_vfmin_vf_f16m##lmul(x, c_exp_hi_half, vl);                 \
        x = __riscv_vfmax_vf_f16m##lmul(x, c_exp_lo_half, vl);                 \
                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                              \
        fx = __riscv_vfmacc_vf_f16m##lmul(                                     \
            __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl), c_cephes_LOG2EF_half, x,  \
            vl);                                                               \
                                                                               \
        /* perform a floorf */                                                 \
        tmp = __riscv_vfcvt_f_x_v_f16m##lmul(                                  \
            __riscv_vfcvt_x_f_v_i16m##lmul(fx, vl), vl);                       \
                                                                               \
        /* if greater, substract 1 */                                          \
        vbool##mlen##_t mask =                                                 \
            __riscv_vmfgt_vv_f16m##lmul##_b##mlen(tmp, fx, vl);                \
        fx = __riscv_vfsub_vf_f16m##lmul##_mu(mask, tmp, tmp, 1.f16, vl);      \
                                                                               \
        tmp = __riscv_vfmul_vf_f16m##lmul(fx, c_cephes_exp_C1_half, vl);       \
        vfloat16m##lmul##_t z =                                                \
            __riscv_vfmul_vf_f16m##lmul(fx, c_cephes_exp_C2_half, vl);         \
        x = __riscv_vfsub_vv_f16m##lmul(x, tmp, vl);                           \
        x = __riscv_vfsub_vv_f16m##lmul(x, z, vl);                             \
                                                                               \
        vfloat16m##lmul##_t y =                                                \
            __riscv_vfmul_vf_f16m##lmul(x, c_cephes_exp_p0_half, vl);          \
        z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);                             \
                                                                               \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_exp_p1_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_exp_p2_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_exp_p3_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_exp_p4_half, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, c_cephes_exp_p5_half, vl);          \
                                                                               \
        y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                             \
        y = __riscv_vfadd_vv_f16m##lmul(y, x, vl);                             \
        y = __riscv_vfadd_vf_f16m##lmul(y, 1.f16, vl);                         \
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
#define _RVV_FLOAT16_SINCOS_OP(LMUL, MLEN)                                     \
    static inline void sincos_ps_fp16(vfloat16m##LMUL##_t x,                   \
                                      vfloat16m##LMUL##_t *ysin,               \
                                      vfloat16m##LMUL##_t *ycos, size_t vl) {  \
        constexpr auto c_minus_cephes_DP1_half = (_Float16)-0.78515625;        \
        constexpr auto c_minus_cephes_DP2_half =                               \
            (_Float16)-2.4187564849853515625e-4;                               \
        constexpr auto c_minus_cephes_DP3_half =                               \
            (_Float16)-3.77489497744594108e-8;                                 \
        constexpr auto c_sincof_p0_half = (_Float16)-1.9515295891E-4;          \
        constexpr auto c_sincof_p1_half = (_Float16)8.3321608736E-3;           \
        constexpr auto c_sincof_p2_half = (_Float16)-1.6666654611E-1;          \
        constexpr auto c_coscof_p0_half = (_Float16)2.443315711809948E-005;    \
        constexpr auto c_coscof_p1_half = (_Float16)-1.388731625493765E-003;   \
        constexpr auto c_coscof_p2_half = (_Float16)4.166664568298827E-002;    \
        constexpr auto c_cephes_FOPI_half = (_Float16)1.27323954473516;        \
        /* any x */                                                            \
        vfloat16m##LMUL##_t xmm1, xmm2, xmm3, y;                               \
                                                                               \
        vuint16m##LMUL##_t emm2;                                               \
                                                                               \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                          \
        sign_mask_sin = __riscv_vmflt_vf_f16m##LMUL##_b##MLEN(x, 0.f16, vl);   \
        x = __riscv_vfsgnj_vf_f16m##LMUL(x, 1.f16, vl);                        \
                                                                               \
        /* scale by 4/Pi */                                                    \
        y = __riscv_vfmul_vf_f16m##LMUL(x, c_cephes_FOPI_half, vl);            \
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
        xmm1 = __riscv_vfmul_vf_f16m##LMUL(y, c_minus_cephes_DP1_half, vl);    \
        xmm2 = __riscv_vfmul_vf_f16m##LMUL(y, c_minus_cephes_DP2_half, vl);    \
        xmm3 = __riscv_vfmul_vf_f16m##LMUL(y, c_minus_cephes_DP3_half, vl);    \
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
        y1 = __riscv_vfmul_vf_f16m##LMUL(z, c_coscof_p0_half, vl);             \
        y2 = __riscv_vfmul_vf_f16m##LMUL(z, c_sincof_p0_half, vl);             \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, c_coscof_p1_half, vl);            \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, c_sincof_p1_half, vl);            \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, c_coscof_p2_half, vl);            \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, c_sincof_p2_half, vl);            \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                           \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                           \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfsub_vv_f16m##LMUL(                                      \
            y1, __riscv_vfmul_vf_f16m##LMUL(z, 0.5f16, vl), vl);               \
        y2 = __riscv_vfadd_vv_f16m##LMUL(y2, x, vl);                           \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, 1.f16, vl);                       \
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
#define _RVV_FLOAT16_TANH_OP(LMUL, MLEN)                                       \
    static inline vfloat16m##LMUL##_t tanh_ps_fp16(vfloat16m##LMUL##_t v,      \
                                                   size_t vl) {                \
        constexpr _Float16 log2_inv = (_Float16)(0x1.715476p+0f);              \
        constexpr _Float16 log2_hi = (_Float16)(0x1.62E42p-1f);                \
        constexpr _Float16 log2_lo = (_Float16)(0x1.ABC9Ep-56f);               \
        constexpr _Float16 fp_posZero = (_Float16)(0.0f);                      \
        constexpr _Float16 fp_posOne = (_Float16)(1.0f);                       \
        constexpr _Float16 range_limit = (_Float16)(9.8f);                     \
        constexpr _Float16 neg_two = (_Float16)(-2.0f);                        \
        constexpr _Float16 eps = (_Float16)(0x1p-10f);                         \
                                                                               \
        constexpr _Float16 c0 = (_Float16)(0x1.6F3A8p-19f);                    \
        constexpr _Float16 c1 = (_Float16)(0x1.9E09Ep-13f);                    \
        constexpr _Float16 c2 = (_Float16)(0x1.0E38Ep-7f);                     \
        constexpr _Float16 c3 = (_Float16)(0x1.5555p-3f);                      \
        constexpr _Float16 c4 = (_Float16)(0x1.9F01Ap-16f);                    \
        constexpr _Float16 c5 = (_Float16)(0x1.6B0C7p-10f);                    \
        constexpr _Float16 c6 = (_Float16)(0x1.4E5E6p-5f);                     \
        constexpr _Float16 c7 = (_Float16)(0x1.6F3A8p-19f);                    \
        constexpr _Float16 c8 = (_Float16)(0x1.9F01Ap-16f);                    \
                                                                               \
        auto c0_vec = __riscv_vfmv_v_f_f16m##LMUL(c0, vl);                     \
        auto c1_vec = __riscv_vfmv_v_f_f16m##LMUL(c1, vl);                     \
        auto c2_vec = __riscv_vfmv_v_f_f16m##LMUL(c2, vl);                     \
        auto c3_vec = __riscv_vfmv_v_f_f16m##LMUL(c3, vl);                     \
        auto c4_vec = __riscv_vfmv_v_f_f16m##LMUL(c4, vl);                     \
        auto c5_vec = __riscv_vfmv_v_f_f16m##LMUL(c5, vl);                     \
        auto c6_vec = __riscv_vfmv_v_f_f16m##LMUL(c6, vl);                     \
                                                                               \
        auto zero = __riscv_vfmv_v_f_f16m##LMUL(fp_posZero, vl);               \
        auto one = __riscv_vfmv_v_f_f16m##LMUL(fp_posOne, vl);                 \
                                                                               \
        auto vx = __riscv_vfsgnj_vf_f16m##LMUL(v, fp_posOne, vl);              \
        vx = __riscv_vfmin_vf_f16m##LMUL(vx, range_limit, vl);                 \
                                                                               \
        vx = __riscv_vfmul_vf_f16m##LMUL(vx, neg_two, vl);                     \
        auto n_flt = __riscv_vfmul_vf_f16m##LMUL(vx, log2_inv, vl);            \
        auto n = __riscv_vfcvt_x_f_v_i16m##LMUL(n_flt, vl);                    \
        auto n_flt_from_n = __riscv_vfcvt_f_x_v_f16m##LMUL(n, vl);             \
                                                                               \
        auto r_delta_hi =                                                      \
            __riscv_vfnmsac_vf_f16m##LMUL(vx, log2_hi, n_flt_from_n, vl);      \
        auto r_delta_lo = __riscv_vfmul_vf_f16m##LMUL(n_flt_from_n, eps, vl);  \
        auto r_delta =                                                         \
            __riscv_vfadd_vv_f16m##LMUL(r_delta_hi, r_delta_lo, vl);           \
                                                                               \
        auto u = __riscv_vadd_vx_i16m##LMUL(n, 15, vl);                        \
        u = __riscv_vsll_vx_i16m##LMUL(u, 10, vl);                             \
        auto r =                                                               \
            __riscv_vfnmsac_vf_f16m##LMUL(r_delta, log2_lo, n_flt_from_n, vl); \
        auto s = __riscv_vreinterpret_v_i16m##LMUL##_f16m##LMUL(u);            \
                                                                               \
        auto s_is_small = __riscv_vmsle_vx_i16m##LMUL##_b##MLEN(n, -11, vl);   \
        r_delta = __riscv_vfsub_vv_f16m##LMUL(r_delta, r, vl);                 \
        auto s_head =                                                          \
            __riscv_vfmerge_vfm_f16m##LMUL(s, fp_posZero, s_is_small, vl);     \
        r_delta =                                                              \
            __riscv_vfnmsac_vf_f16m##LMUL(r_delta, log2_lo, n_flt_from_n, vl); \
                                                                               \
        auto rsq = __riscv_vfmul_vv_f16m##LMUL(r, r, vl);                      \
        auto s_tail = __riscv_vmerge_vvm_f16m##LMUL(zero, s, s_is_small, vl);  \
        auto rcube = __riscv_vfmul_vv_f16m##LMUL(rsq, r, vl);                  \
                                                                               \
        auto p_even = __riscv_vfmadd_vf_f16m##LMUL(rsq, c7, c0_vec, vl);       \
        p_even = __riscv_vfmadd_vv_f16m##LMUL(p_even, rsq, c1_vec, vl);        \
        p_even = __riscv_vfmadd_vv_f16m##LMUL(p_even, rsq, c2_vec, vl);        \
        p_even = __riscv_vfmadd_vv_f16m##LMUL(p_even, rsq, c3_vec, vl);        \
                                                                               \
        auto p_odd = __riscv_vfmadd_vf_f16m##LMUL(rsq, c8, c4_vec, vl);        \
        p_odd = __riscv_vfmadd_vv_f16m##LMUL(p_odd, rsq, c5_vec, vl);          \
        p_odd = __riscv_vfmadd_vv_f16m##LMUL(p_odd, rsq, c6_vec, vl);          \
                                                                               \
        auto poly = __riscv_vfmadd_vv_f16m##LMUL(p_odd, r, p_even, vl);        \
                                                                               \
        auto r_prime = __riscv_vfmul_vf_f16m##LMUL(r, 0.5f16, vl);             \
        auto B = __riscv_vfmadd_vv_f16m##LMUL(r, r_prime, r, vl);              \
        auto b = __riscv_vfsub_vv_f16m##LMUL(r, B, vl);                        \
        b = __riscv_vfmacc_vv_f16m##LMUL(b, r, r_prime, vl);                   \
        auto c = __riscv_vfmadd_vv_f16m##LMUL(r, r_delta, r_delta, vl);        \
        b = __riscv_vfadd_vv_f16m##LMUL(b, c, vl);                             \
        poly = __riscv_vfmadd_vv_f16m##LMUL(poly, rcube, b, vl);               \
                                                                               \
        auto Z = __riscv_vfadd_vf_f16m##LMUL(s_head, fp_posOne, vl);           \
        auto D_tmp = __riscv_vfmadd_vv_f16m##LMUL(B, s, Z, vl);                \
        auto d_tmp = __riscv_vfsub_vv_f16m##LMUL(Z, D_tmp, vl);                \
        d_tmp = __riscv_vfmacc_vv_f16m##LMUL(d_tmp, s, B, vl);                 \
        d_tmp = __riscv_vfadd_vv_f16m##LMUL(d_tmp, s_tail, vl);                \
        d_tmp = __riscv_vfmacc_vv_f16m##LMUL(d_tmp, s, poly, vl);              \
                                                                               \
        auto D = __riscv_vfadd_vv_f16m##LMUL(D_tmp, d_tmp, vl);                \
        auto d = __riscv_vfsub_vv_f16m##LMUL(D_tmp, D, vl);                    \
        d = __riscv_vfadd_vv_f16m##LMUL(d, d_tmp, vl);                         \
                                                                               \
        auto E = __riscv_vfrdiv_vf_f16m##LMUL(D, fp_posOne, vl);               \
        auto e1 = __riscv_vfnmsub_vv_f16m##LMUL(E, D, one, vl);                \
        E = __riscv_vfmadd_vv_f16m##LMUL(E, e1, E, vl);                        \
        auto e2 = __riscv_vfnmsub_vv_f16m##LMUL(E, D, one, vl);                \
        E = __riscv_vfmadd_vv_f16m##LMUL(E, e2, E, vl);                        \
                                                                               \
        Z = __riscv_vfrsub_vf_f16m##LMUL(s_head, fp_posOne, vl);               \
        auto Numer = __riscv_vfnmsub_vv_f16m##LMUL(B, s, Z, vl);               \
        auto numer = __riscv_vfsub_vv_f16m##LMUL(Z, Numer, vl);                \
        numer = __riscv_vfnmsac_vv_f16m##LMUL(numer, s, B, vl);                \
        numer = __riscv_vfsub_vv_f16m##LMUL(numer, s_tail, vl);                \
        numer = __riscv_vfnmsac_vv_f16m##LMUL(numer, s, poly, vl);             \
                                                                               \
        auto vy = __riscv_vfmul_vv_f16m##LMUL(E, numer, vl);                   \
        vy = __riscv_vfmacc_vv_f16m##LMUL(vy, Numer, E, vl);                   \
        vy = __riscv_vfmacc_vv_f16m##LMUL(vy, numer, E, vl);                   \
        vy = __riscv_vfmacc_vv_f16m##LMUL(vy, Numer, E, vl);                   \
                                                                               \
        return __riscv_vfsgnj_vv_f16m##LMUL(vy, v, vl);                        \
    }

_RVV_FLOAT16_TANH_OP(1, 16)
_RVV_FLOAT16_TANH_OP(2, 8)
_RVV_FLOAT16_TANH_OP(4, 4)
_RVV_FLOAT16_TANH_OP(8, 2)

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
struct sv_erff_half_data {
    static constexpr size_t N = 513;
    _Float16 erf[N];
    _Float16 scale[N];
    
    sv_erff_half_data(const sv_erff_data& src) {
        for(size_t i=0; i<N; ++i) {
            erf[i] = static_cast<_Float16>(src.erf[i]);
            scale[i] = static_cast<_Float16>(src.scale[i]);
        }
    }
};

const sv_erff_data& erf_data_fp32 = __sv_erff_data; 
const sv_erff_half_data erf_data_half(erf_data_fp32);

#define _RVV_FLOAT16_ERF_OP(LMUL, MLEN, TLEN)                                  \
    static inline vfloat##TLEN##m##LMUL##_t erf_ps_fp16(                       \
        vfloat##TLEN##m##LMUL##_t x, size_t vl) {                              \
        constexpr auto c0 = (_Float16)(0x1.cp-7f);                             \
        constexpr auto c1 = (_Float16)(3.9375f);                               \
        constexpr auto c2 = (_Float16)(0x1.555556p-2f);                        \
        auto zero = __riscv_vmv_v_x_u16m##LMUL(0, vl);                         \
        auto a = __riscv_vfabs_v_f##TLEN##m##LMUL(x, vl);                      \
                                                                               \
        /* |x| > 1/64 - 1/512. */                                              \
        auto gt_min_mask =                                                     \
            __riscv_vmfgt_vf_f##TLEN##m##LMUL##_b##MLEN(a, c0, vl);            \
                                                                               \
        auto tmp_i = __riscv_vfmul_vf_f##TLEN##m##LMUL(a, 128.f16, vl);        \
        auto i = __riscv_vfcvt_xu_f_v_u16m##LMUL(tmp_i, vl);                   \
                                                                               \
        /* Saturate lookup index. */                                           \
        i = __riscv_vmerge_vvm_u16m##LMUL(zero, i, gt_min_mask, vl);           \
        i = __riscv_vminu_vx_u16m##LMUL(i, 512, vl);                           \
        auto tmp_r = __riscv_vfcvt_f_xu_v_f##TLEN##m##LMUL(i, vl);             \
        auto r = __riscv_vfmul_vf_f##TLEN##m##LMUL(tmp_r,                      \
                                                   (_Float16)(1.f / 128), vl); \
        auto byte_i = __riscv_vmul_vx_u16m##LMUL(i, 2, vl); /* byte offset */  \
                                                                               \
        /* r and erf(r) set to 0 for |x| below min. */                         \
        auto erfr = __riscv_vluxei16_v_f##TLEN##m##LMUL(erf_data_half.erf,     \
                                                        byte_i, vl);           \
        auto scale = __riscv_vluxei16_v_f##TLEN##m##LMUL(erf_data_half.scale,  \
                                                         byte_i, vl);          \
                                                                               \
        /* |x| >= 4.0 - 8/128. */                                              \
        auto ge_max_mask =                                                     \
            __riscv_vmfge_vf_f##TLEN##m##LMUL##_b##MLEN(a, c1, vl);            \
                                                                               \
        /* erf(x) ~ erf(r) + scale * d * (1 - r * d - 1/3 * d^2). */           \
        auto d = __riscv_vfsub_vv_f##TLEN##m##LMUL(a, r, vl);                  \
        auto d2 = __riscv_vfmul_vv_f##TLEN##m##LMUL(d, d, vl);                 \
        auto y = __riscv_vfmacc_vf_f##TLEN##m##LMUL(r, c2, d, vl);             \
        y = __riscv_vfnmsub_vv_f##TLEN##m##LMUL(y, d2, d, vl);                 \
        y = __riscv_vfmadd_vv_f##TLEN##m##LMUL(y, scale, erfr, vl);            \
                                                                               \
        /* Solves the |x| = inf case. */                                       \
        y = __riscv_vfmerge_vfm_f##TLEN##m##LMUL(y, 1.f16, ge_max_mask, vl);   \
                                                                               \
        /* Copy sign. */                                                       \
        return __riscv_vfsgnj_vv_f##TLEN##m##LMUL(y, x, vl);                   \
    }

_RVV_FLOAT16_ERF_OP(1, 16, 16)
_RVV_FLOAT16_ERF_OP(2, 8, 16)
_RVV_FLOAT16_ERF_OP(4, 4, 16)
_RVV_FLOAT16_ERF_OP(8, 2, 16)

#endif