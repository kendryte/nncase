

#pragma once 
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/vector.h"
#include "../../../half.h"
#include "rvv_mathfun.h"
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif


namespace nncase::ntt::ops{

//note: mlen should be fix(1,32)->(1,16)
#ifndef REGISTER_RVV_FP16_KERNEL
#define REGISTER_RVV_FP16_KERNEL(kernel)                                            \
    kernel(1, 16) kernel(2, 8) kernel(4, 4) kernel(8, 2)
#endif

#define RVV_UNARY16_OP(op, dtype, vl, kernel)                                    \
    template <> struct op<ntt::vector<dtype, vl>> {                            \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

// unary with hlaf
#define REGISTER_RVV_UNARY16_OP(OP, dtype, kernel)                               \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)           \
    RVV_UNARY16_OP(OP, half, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

#define ABS_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t abs_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {  \
         return __riscv_vfabs_v_f16m##lmul(v, vl);   \
    }

REGISTER_RVV_FP16_KERNEL(ABS_FLOAT16)
REGISTER_RVV_UNARY16_OP(abs, half, abs_float16)

#define ACOS_FLOAT16(lmul, mlen) \
    inline vfloat16m##lmul##_t acos_float16(const vfloat16m##lmul##_t& v,\
                                            const size_t vl){\
        auto zero = __riscv_vfmv_v_f_f16m##lmul(0.f16, vl);                        \
        auto half = __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl);                     \
        auto one = __riscv_vfmv_v_f_f16m##lmul(1.f16, vl);                       \
        auto two = __riscv_vfmv_v_f_f16m##lmul(2.f16, vl);                       \
        auto minus_one = __riscv_vfmv_v_f_f16m##lmul(-1.f16, vl);                \
        auto p0 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.55555ep-3), vl);              \
        auto p1 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.33261ap-4), vl);              \
        auto p2 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.70d7dcp-5), vl);              \
        auto neg_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);     \
        auto x = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        auto off = __riscv_vfmerge_vfm_f16m##lmul(zero, static_cast<_Float16>(0x1.921fb6p+1f),        \
                                                  neg_mask, vl);               \
        auto mul1 = __riscv_vfmerge_vfm_f16m##lmul(two, -2.f16, neg_mask, vl);   \
        auto mul2 =                                                            \
            __riscv_vfmerge_vfm_f16m##lmul(minus_one, 1.f16, neg_mask, vl);      \
        auto le_half_mask =                                                    \
            __riscv_vmfle_vv_f16m##lmul##_b##mlen(x, half, vl);                \
        auto tmp = __riscv_vmv_v_v_f16m##lmul(x, vl);                          \
        auto mul =                                                             \
            __riscv_vmerge_vvm_f16m##lmul(mul1, mul2, le_half_mask, vl);       \
        tmp = __riscv_vfnmsub_vv_f16m##lmul(tmp, half, half, vl);              \
        auto v2 = __riscv_vfmul_vv_f16m##lmul(v, v, vl);                       \
        auto add = __riscv_vfmerge_vfm_f16m##lmul(off, static_cast<_Float16>(0x1.921fb6p+0f),         \
                                                  le_half_mask, vl);           \
        auto z2 = __riscv_vmerge_vvm_f16m##lmul(tmp, v2, le_half_mask, vl);    \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.3af7d8p-5), vl);              \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.b059dp-6), vl);               \
        tmp = __riscv_vfsqrt_v_f16m##lmul(z2, vl);                             \
        auto z4 = __riscv_vfmul_vv_f16m##lmul(z2, z2, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p2, vl);                     \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, z4, p1, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p0, vl);                     \
        auto z = __riscv_vmerge_vvm_f16m##lmul(tmp, x, le_half_mask, vl);      \
        y1 = __riscv_vfmacc_vv_f16m##lmul(y1, y2, z2, vl);                     \
        mul = __riscv_vfmul_vv_f16m##lmul(mul, z, vl);                         \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z2, one, vl);                    \
        return __riscv_vfmadd_vv_f16m##lmul(y1, mul, add, vl);              \
        }  
REGISTER_RVV_FP16_KERNEL(ACOS_FLOAT16)
REGISTER_RVV_UNARY16_OP(acos, half, acos_float16)


#define ASIN_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t asin_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto half = __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl);                     \
        auto one = __riscv_vfmv_v_f_f16m##lmul(1.f16, vl);                       \
        auto minus_two = __riscv_vfmv_v_f_f16m##lmul(-2.f16, vl);                \
        auto pi_over_2f = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.921fb6p+0f), vl);     \
        auto p0 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.55555ep-3), vl);              \
        auto p1 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.31661ap-4), vl);              \
        auto p2 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.70d7dcp-5), vl);              \
        auto neg_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);     \
        auto x = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        auto mul1 = __riscv_vfmerge_vfm_f16m##lmul(one, -1.f16, neg_mask, vl);   \
                                                                               \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto lt_half_mask =                                                    \
            __riscv_vmflt_vv_f16m##lmul##_b##mlen(x, half, vl);                \
        auto tmp = __riscv_vmv_v_v_f16m##lmul(x, vl);                          \
        auto mul2 =                                                            \
            __riscv_vfmerge_vfm_f16m##lmul(minus_two, 1.f16, lt_half_mask, vl);  \
        tmp = __riscv_vfnmsub_vv_f16m##lmul(tmp, half, half, vl);              \
        auto add =                                                             \
            __riscv_vfmerge_vfm_f16m##lmul(pi_over_2f, 0.f16, lt_half_mask, vl); \
        auto v2 = __riscv_vfmul_vv_f16m##lmul(v, v, vl);                       \
        auto z2 = __riscv_vmerge_vvm_f16m##lmul(tmp, v2, lt_half_mask, vl);    \
        /* asin(|x|) = Q(|x|),        for |x| < 0.5                            \
                = pi / 2 - 2 Q(|x|) , for |x| >= 0.5.  */                      \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.3af7d8p-5), vl);              \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(static_cast<_Float16>(0x1.b059dp-6), vl);               \
        auto z4 = __riscv_vfmul_vv_f16m##lmul(z2, z2, vl);                     \
        tmp = __riscv_vfsqrt_v_f16m##lmul(z2, vl);                             \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p2, vl);                     \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, z4, p1, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p0, vl);                     \
        auto z = __riscv_vmerge_vvm_f16m##lmul(tmp, x, lt_half_mask, vl);      \
        y1 = __riscv_vfmacc_vv_f16m##lmul(y1, y2, z2, vl);                     \
        z2 = __riscv_vfmul_vv_f16m##lmul(z2, z, vl);                           \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z2, z, vl);                      \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, mul2, add, vl);                  \
        return __riscv_vfmul_vv_f16m##lmul(y1, mul1, vl);                      \
    }
REGISTER_RVV_FP16_KERNEL(ASIN_FLOAT16)
REGISTER_RVV_UNARY16_OP(asin, half, asin_float16)

// correct
#define CEIL_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t ceil_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto vi = __riscv_vfcvt_x_f_v_i16m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f16m##lmul(vi, vl);                      \
        auto mask = __riscv_vmflt_vv_f16m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfadd_vf_f16m##lmul##_m(mask, vf, 1.f16, vl);               \
        return vf;                                                             \
    }
REGISTER_RVV_FP16_KERNEL(CEIL_FLOAT16)
REGISTER_RVV_UNARY16_OP(ceil, half, ceil_float16)


#define FLOOR_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t floor_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto vi = __riscv_vfcvt_x_f_v_i16m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f16m##lmul(vi, vl);                      \
        auto mask = __riscv_vmfgt_vv_f16m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfsub_vf_f16m##lmul##_m(mask, vf, 1.f16, vl);               \
        return vf;                                                             \
    }
REGISTER_RVV_FP16_KERNEL(FLOOR_FLOAT16)
REGISTER_RVV_UNARY16_OP(floor, half, floor_float16)

#define NEG_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t neg_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return __riscv_vfneg_v_f16m##lmul(v, vl);                              \
    }
REGISTER_RVV_FP16_KERNEL(NEG_FLOAT16)
REGISTER_RVV_UNARY16_OP(neg, half, neg_float16)


#define ROUND_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t round_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        return __riscv_vfcvt_f_x_v_f16m##lmul(                                 \
            __riscv_vfcvt_x_f_v_i16m##lmul(v, vl), vl);                        \
    }
REGISTER_RVV_FP16_KERNEL(ROUND_FLOAT16)
REGISTER_RVV_UNARY16_OP(round, half, round_float16)


#define RSQRT_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t rsqrt_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto one_point_five = __riscv_vfmv_v_f_f16m##lmul(1.5f, vl);           \
                                                                               \
        auto ux = __riscv_vreinterpret_v_f16m##lmul##_u16m##lmul(v);           \
        ux = __riscv_vsrl_vx_u16m##lmul(ux, 1, vl);                            \
        ux = __riscv_vrsub_vx_u16m##lmul(ux, 0x5f375a86, vl);                  \
        auto y = __riscv_vreinterpret_v_u16m##lmul##_f16m##lmul(ux);           \
                                                                               \
        auto y2 = __riscv_vfmul_vv_f16m##lmul(y, y, vl);                       \
        auto x = __riscv_vfmul_vf_f16m##lmul(v, -0.5f, vl);                    \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f16m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f16m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f16m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f16m##lmul(y, y2, vl);                            \
        return y;                                                              \
    }
REGISTER_RVV_FP16_KERNEL(RSQRT_FLOAT16)
REGISTER_RVV_UNARY16_OP(rsqrt, half, rsqrt_float16)


#define SIGN_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t sign_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto ret = __riscv_vfmv_v_f_f16m##lmul(0.f, vl);                       \
        auto gt_mask = __riscv_vmfgt_vf_f16m##lmul##_b##mlen(v, 0.f, vl);      \
        ret = __riscv_vfmerge_vfm_f16m##lmul(ret, 1.f, gt_mask, vl);           \
        auto lt_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f, vl);      \
        return __riscv_vfmerge_vfm_f16m##lmul(ret, -1.f, lt_mask, vl);         \
    }

REGISTER_RVV_FP16_KERNEL(SIGN_FLOAT16)
REGISTER_RVV_UNARY16_OP(sign, half, sign_float16)

#define SQRT_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t sqrt_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        return __riscv_vfsqrt_v_f16m##lmul(v, vl);                             \
    }
REGISTER_RVV_FP16_KERNEL(SQRT_FLOAT16)
REGISTER_RVV_UNARY16_OP(sqrt, half, sqrt_float16)


#define SQUARE_FLOAT16(lmul, mlen)                                             \
    inline vfloat16m##lmul##_t square_float16(const vfloat16m##lmul##_t &v,    \
                                              const size_t vl) {               \
        return __riscv_vfmul_vv_f16m##lmul(v, v, vl);                          \
    }
REGISTER_RVV_FP16_KERNEL(SQUARE_FLOAT16)
REGISTER_RVV_UNARY16_OP(square, half, square_float16)


}
