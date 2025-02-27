#pragma once
#include <cmath>

#if __riscv_vector
#include <riscv_vector.h>

#define c_inv_mant_mask_f16 -31745
#define c_cephes_SQRTHF     0.707106781186547524
#define c_cephes_log_p0     7.0376836292E-2
#define c_cephes_log_p1     -1.1514610310E-1
#define c_cephes_log_p2     1.1676998740E-1
#define c_cephes_log_p3     -1.2420140846E-1
#define c_cephes_log_p4     +1.4249322787E-1
#define c_cephes_log_p5     -1.6668057665E-1
#define c_cephes_log_p6     +2.0000714765E-1
#define c_cephes_log_p7     -2.4999993993E-1
#define c_cephes_log_p8     +3.3333331174E-1
#define c_cephes_log_q1     -2.12194440e-4
#define c_cephes_log_q2     0.693359375



#define LOG_FLOAT16(lmul, mlen)                                             \
    inline vfloat16m##lmul##_t log_ps_fp16(vfloat16m##lmul##_t x, size_t vl)                                                    \
        {                                                                                                                             \
            x = __riscv_vfmax_vf_f16m##lmul(x, (_Float16)0.f, vl); /* force flush to zero on denormal values */                         \
            vbool##mlen##_t invalid_mask = __riscv_vmfle_vf_f16m##lmul##_b##mlen(x, (_Float16)0.f, vl);                                 \
                                                                                                                                    \
            vint16m##lmul##_t ux = __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(x);                                                 \
                                                                                                                                    \
            vint16m##lmul##_t emm0 = __riscv_vsra_vx_i16m##lmul(ux, 10, vl);                                                          \
                                                                                                                                    \
            /* keep only the fractional part */                                                                                       \
            ux = __riscv_vand_vx_i16m##lmul(ux, c_inv_mant_mask_f16, vl);                                                             \
            ux = __riscv_vor_vx_i16m##lmul(ux, 14336 /* reinterpret_cast<short>((_Float16)0.5) */, vl);                                 \
            x = __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(ux);                                                                   \
                                                                                                                                    \
            emm0 = __riscv_vsub_vx_i16m##lmul(emm0, 0xf, vl);                                                                         \
            vfloat16m##lmul##_t e = __riscv_vfcvt_f_x_v_f16m##lmul(emm0, vl);                                                         \
                                                                                                                                    \
            e = __riscv_vfadd_vf_f16m##lmul(e, (_Float16)1.f, vl);                                                                      \
                                                                                                                                    \
            /* part2:                      */                                                                                         \
            /*     if( x < SQRTHF ) {      */                                                                                         \
            /*       e -= 1;               */                                                                                         \
            /*       x = x + x - 1.0;      */                                                                                         \
            /*     } else { x = x - 1.0; } */                                                                                         \
            vbool##mlen##_t mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(x, (_Float16)c_cephes_SQRTHF, vl);                             \
            x = __riscv_vfadd_vv_f16m##lmul##_mu(mask, x, x, x, vl);                                                                  \
            x = __riscv_vfsub_vf_f16m##lmul(x, (_Float16)1.f, vl);                                                                      \
            e = __riscv_vfsub_vf_f16m##lmul##_mu(mask, e, e, (_Float16)1.f, vl);                                                        \
                                                                                                                                    \
            vfloat16m##lmul##_t z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);                                                            \
                                                                                                                                    \
            vfloat16m##lmul##_t y = __riscv_vfmul_vf_f16m##lmul(x, (_Float16)c_cephes_log_p0, vl);                                      \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p1, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p2, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p3, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p4, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p5, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p6, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p7, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
            y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_log_p8, vl);                                                          \
            y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                                \
                                                                                                                                    \
            y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                                                                                \
                                                                                                                                    \
            vfloat16m##lmul##_t tmp = __riscv_vfmul_vf_f16m##lmul(e, (_Float16)c_cephes_log_q1, vl);                                    \
            y = __riscv_vfadd_vv_f16m##lmul(y, tmp, vl);                                                                              \
                                                                                                                                    \
            tmp = __riscv_vfmul_vf_f16m##lmul(z, (_Float16)0.5f, vl);                                                                   \
            y = __riscv_vfsub_vv_f16m##lmul(y, tmp, vl);                                                                              \
                                                                                                                                    \
            tmp = __riscv_vfmul_vf_f16m##lmul(e, (_Float16)c_cephes_log_q2, vl);                                                        \
            x = __riscv_vfadd_vv_f16m##lmul(x, y, vl);                                                                                \
            x = __riscv_vfadd_vv_f16m##lmul(x, tmp, vl);                                                                              \
            /* negative arg will be NAN */                                                                                            \
            vuint16m##lmul##_t xtmp = __riscv_vreinterpret_v_f16m##lmul##_u16m##lmul(x);                                              \
            x = __riscv_vreinterpret_v_u16m##lmul##_f16m##lmul(__riscv_vor_vx_u16m##lmul##_mu(invalid_mask, xtmp, xtmp, 0xffff, vl)); \
            return x;                                                                                                                 \
        }

LOG_FLOAT16(1, 16)
LOG_FLOAT16(2, 8)
LOG_FLOAT16(4, 4)
LOG_FLOAT16(8, 2)

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


#define EXP_FLOAT16(lmul, mlen)                                                                                   \
    static inline vfloat16m##lmul##_t exp_ps_fp16(vfloat16m##lmul##_t x, size_t vl)                                            \
    {                                                                                                                     \
        vfloat16m##lmul##_t tmp, fx;                                                                                      \
                                                                                                                          \
        x = __riscv_vfmin_vf_f16m##lmul(x, (_Float16)c_exp_hi_f16, vl);                                                     \
        x = __riscv_vfmax_vf_f16m##lmul(x, (_Float16)c_exp_lo_f16, vl);                                                     \
                                                                                                                          \
        /* express exp(x) as exp(g + n*log(2)) */                                                                         \
        fx = __riscv_vfmacc_vf_f16m##lmul(__riscv_vfmv_v_f_f16m##lmul((_Float16)0.5f, vl), (_Float16)c_cephes_LOG2EF, x, vl); \
                                                                                                                          \
        /* perform a floorf */                                                                                            \
        tmp = __riscv_vfcvt_f_x_v_f16m##lmul(__riscv_vfcvt_x_f_v_i16m##lmul(fx, vl), vl);                                 \
                                                                                                                          \
        /* if greater, substract 1 */                                                                                     \
        vbool##mlen##_t mask = __riscv_vmfgt_vv_f16m##lmul##_b##mlen(tmp, fx, vl);                                        \
        fx = __riscv_vfsub_vf_f16m##lmul##_mu(mask, tmp, tmp, (_Float16)1.f, vl);                                           \
                                                                                                                          \
        tmp = __riscv_vfmul_vf_f16m##lmul(fx, (_Float16)c_cephes_exp_C1, vl);                                               \
        vfloat16m##lmul##_t z = __riscv_vfmul_vf_f16m##lmul(fx, (_Float16)c_cephes_exp_C2, vl);                             \
        x = __riscv_vfsub_vv_f16m##lmul(x, tmp, vl);                                                                      \
        x = __riscv_vfsub_vv_f16m##lmul(x, z, vl);                                                                        \
                                                                                                                          \
        vfloat16m##lmul##_t y = __riscv_vfmul_vf_f16m##lmul(x, (_Float16)c_cephes_exp_p0, vl);                              \
        z = __riscv_vfmul_vv_f16m##lmul(x, x, vl);                                                                        \
                                                                                                                          \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_exp_p1, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_exp_p2, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_exp_p3, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_exp_p4, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##lmul(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)c_cephes_exp_p5, vl);                                                  \
                                                                                                                          \
        y = __riscv_vfmul_vv_f16m##lmul(y, z, vl);                                                                        \
        y = __riscv_vfadd_vv_f16m##lmul(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##lmul(y, (_Float16)1.f, vl);                                                              \
                                                                                                                          \
        /* build 2^n */                                                                                                   \
        vint16m##lmul##_t mm = __riscv_vfcvt_x_f_v_i16m##lmul(fx, vl);                                                    \
        mm = __riscv_vadd_vx_i16m##lmul(mm, 0xf, vl);                                                                     \
        mm = __riscv_vsll_vx_i16m##lmul(mm, 10, vl);                                                                      \
        vfloat16m##lmul##_t pow2n = __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(mm);                                   \
                                                                                                                          \
        y = __riscv_vfmul_vv_f16m##lmul(y, pow2n, vl);                                                                    \
        return y;                                                                                                         \
    }
    
EXP_FLOAT16(1, 16)
EXP_FLOAT16(2, 8)
EXP_FLOAT16(4, 4)
EXP_FLOAT16(8, 2)

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0        -1.9515295891E-4
#define c_sincof_p1        8.3321608736E-3
#define c_sincof_p2        -1.6666654611E-1
#define c_coscof_p0        2.443315711809948E-005
#define c_coscof_p1        -1.388731625493765E-003
#define c_coscof_p2        4.166664568298827E-002
#define c_cephes_FOPI      1.27323954473516 // 4 / M_PI

#define SINCOS_FLOAT16(LMUL, MLEN)                                                                                                                  \
    static inline void sincos_ps_fp16(vfloat16m##LMUL##_t x, vfloat16m##LMUL##_t* ysin, vfloat16m##LMUL##_t* ycos, size_t vl)                                    \
    {                                                                                                                                                       \
        /* any x */                                                                                                                                         \
        vfloat16m##LMUL##_t xmm1, xmm2, xmm3, y;                                                                                                            \
                                                                                                                                                            \
        vuint16m##LMUL##_t emm2;                                                                                                                            \
                                                                                                                                                            \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                                                                                                       \
        sign_mask_sin = __riscv_vmflt_vf_f16m##LMUL##_b##MLEN(x, (_Float16)0.f, vl);                                                                          \
        x = __riscv_vfsgnj_vf_f16m##LMUL(x, (_Float16)1.f, vl);                                                                                               \
                                                                                                                                                            \
        /* scale by 4/Pi */                                                                                                                                 \
        y = __riscv_vfmul_vf_f16m##LMUL(x, (_Float16)c_cephes_FOPI, vl);                                                                                      \
                                                                                                                                                            \
        /* store the integer part of y in mm0 */                                                                                                            \
        emm2 = __riscv_vfcvt_xu_f_v_u16m##LMUL(y, vl);                                                                                                      \
        /* j=(j+1) & (~1) (see the cephes sources) */                                                                                                       \
        emm2 = __riscv_vadd_vx_u16m##LMUL(emm2, 1, vl);                                                                                                     \
        emm2 = __riscv_vand_vx_u16m##LMUL(emm2, ~1, vl);                                                                                                    \
        y = __riscv_vfcvt_f_xu_v_f16m##LMUL(emm2, vl);                                                                                                      \
                                                                                                                                                            \
        /* get the polynom selection mask              */                                                                                                   \
        /*     there is one polynom for 0 <= x <= Pi/4 */                                                                                                   \
        /*     and another one for Pi/4<x<=Pi/2        */                                                                                                   \
        /*                                             */                                                                                                   \
        /*     Both branches will be computed.         */                                                                                                   \
        vbool##MLEN##_t poly_mask = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(emm2, 2, vl), 0, vl);                                  \
                                                                                                                                                            \
        /* The magic pass: "Extended precision modular arithmetic" */                                                                                       \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */                                                                                       \
        xmm1 = __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP1, vl);                                                                              \
        xmm2 = __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP2, vl);                                                                              \
        xmm3 = __riscv_vfmul_vf_f16m##LMUL(y, (_Float16)c_minus_cephes_DP3, vl);                                                                              \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm1, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm2, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm3, vl);                                                                                                       \
                                                                                                                                                            \
        sign_mask_sin = __riscv_vmxor_mm_b##MLEN(sign_mask_sin, __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(emm2, 4, vl), 0, vl), vl); \
        sign_mask_cos = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(__riscv_vsub_vx_u16m##LMUL(emm2, 2, vl), 4, vl), 0, vl);           \
                                                                                                                                                            \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */                                                                                           \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */                                                                                           \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vv_f16m##LMUL(x, x, vl);                                                                                      \
        vfloat16m##LMUL##_t y1, y2;                                                                                                                         \
                                                                                                                                                            \
        y1 = __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)c_coscof_p0, vl);                                                                                       \
        y2 = __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)c_sincof_p0, vl);                                                                                       \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)c_coscof_p1, vl);                                                                                      \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (_Float16)c_sincof_p1, vl);                                                                                      \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)c_coscof_p2, vl);                                                                                      \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (_Float16)c_sincof_p2, vl);                                                                                      \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfsub_vv_f16m##LMUL(y1, __riscv_vfmul_vf_f16m##LMUL(z, (_Float16)0.5f, vl), vl);                                                         \
        y2 = __riscv_vfadd_vv_f16m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (_Float16)1.f, vl);                                                                                              \
                                                                                                                                                            \
        /* select the correct result from the two polynoms */                                                                                               \
        vfloat16m##LMUL##_t ys = __riscv_vmerge_vvm_f16m##LMUL(y2, y1, poly_mask, vl);                                                                      \
        vfloat16m##LMUL##_t yc = __riscv_vmerge_vvm_f16m##LMUL(y1, y2, poly_mask, vl);                                                                      \
        *ysin = __riscv_vmerge_vvm_f16m##LMUL(ys, __riscv_vfneg_v_f16m##LMUL(ys, vl), sign_mask_sin, vl);                                                   \
        *ycos = __riscv_vmerge_vvm_f16m##LMUL(__riscv_vfneg_v_f16m##LMUL(yc, vl), yc, sign_mask_cos, vl);                                                   \
    }
SINCOS_FLOAT16(1, 16)
SINCOS_FLOAT16(2, 8)
SINCOS_FLOAT16(4, 4)
SINCOS_FLOAT16(8, 2)




#endif