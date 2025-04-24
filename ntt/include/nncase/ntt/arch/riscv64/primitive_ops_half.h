

#pragma once
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/vector.h"
#include "rvv_mathfun.h"
#include "rvv_mathfun_half.h"
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace nncase::ntt::ops {

#ifndef REGISTER_RVV_FP16_KERNEL
#define REGISTER_RVV_FP16_KERNEL(kernel)                                       \
    kernel(1, 16) kernel(2, 8) kernel(4, 4) kernel(8, 2)
#endif

#define RVV_UNARY_FP16_OP(op, dtype, vl, kernel)                               \
    template <> struct op<ntt::vector<dtype, vl>> {                            \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

// unary with _Float16
#define REGISTER_RVV_UNARY_FP16_OP(OP, dtype, kernel)                          \
    RVV_UNARY_FP16_OP(OP, dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)      \
    RVV_UNARY_FP16_OP(OP, dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)      \
    RVV_UNARY_FP16_OP(OP, dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)      \
    RVV_UNARY_FP16_OP(OP, dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

// abs
#define ABS_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t abs_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return __riscv_vfabs_v_f16m##lmul(v, vl);                              \
    }

REGISTER_RVV_FP16_KERNEL(ABS_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(abs, _Float16, abs_float16)

// acos
#define ACOS_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t acos_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        constexpr auto pc0 = (_Float16)(0x1.55555ep-3);                        \
        constexpr auto pc1 = (_Float16)(0x1.33261ap-4);                        \
        constexpr auto pc2 = (_Float16)(0x1.70d7dcp-5);                        \
        constexpr auto pc3 = (_Float16)(0x1.921fb6p+1f);                       \
        constexpr auto pc4 = (_Float16)(0x1.921fb6p+0f);                       \
        constexpr auto pc5 = (_Float16)(0x1.3af7d8p-5);                        \
        constexpr auto pc6 = (_Float16)(0x1.b059dp-6);                         \
        auto zero = __riscv_vfmv_v_f_f16m##lmul(0.f16, vl);                    \
        auto halff = __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl);                  \
        auto one = __riscv_vfmv_v_f_f16m##lmul(1.f16, vl);                     \
        auto two = __riscv_vfmv_v_f_f16m##lmul(2.f16, vl);                     \
        auto minus_one = __riscv_vfmv_v_f_f16m##lmul(-1.f16, vl);              \
        auto p0 = __riscv_vfmv_v_f_f16m##lmul(pc0, vl);                        \
        auto p1 = __riscv_vfmv_v_f_f16m##lmul(pc1, vl);                        \
        auto p2 = __riscv_vfmv_v_f_f16m##lmul(pc2, vl);                        \
        auto neg_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);   \
        auto x = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        auto off = __riscv_vfmerge_vfm_f16m##lmul(zero, pc3, neg_mask, vl);    \
        auto mul1 = __riscv_vfmerge_vfm_f16m##lmul(two, -2.f16, neg_mask, vl); \
        auto mul2 =                                                            \
            __riscv_vfmerge_vfm_f16m##lmul(minus_one, 1.f16, neg_mask, vl);    \
        auto le_half_mask =                                                    \
            __riscv_vmfle_vv_f16m##lmul##_b##mlen(x, halff, vl);               \
        auto tmp = __riscv_vmv_v_v_f16m##lmul(x, vl);                          \
        auto mul =                                                             \
            __riscv_vmerge_vvm_f16m##lmul(mul1, mul2, le_half_mask, vl);       \
        tmp = __riscv_vfnmsub_vv_f16m##lmul(tmp, halff, halff, vl);            \
        auto v2 = __riscv_vfmul_vv_f16m##lmul(v, v, vl);                       \
        auto add = __riscv_vfmerge_vfm_f16m##lmul(off, pc4, le_half_mask, vl); \
        auto z2 = __riscv_vmerge_vvm_f16m##lmul(tmp, v2, le_half_mask, vl);    \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(pc5, vl);                        \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(pc6, vl);                        \
        tmp = __riscv_vfsqrt_v_f16m##lmul(z2, vl);                             \
        auto z4 = __riscv_vfmul_vv_f16m##lmul(z2, z2, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p2, vl);                     \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, z4, p1, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z4, p0, vl);                     \
        auto z = __riscv_vmerge_vvm_f16m##lmul(tmp, x, le_half_mask, vl);      \
        y1 = __riscv_vfmacc_vv_f16m##lmul(y1, y2, z2, vl);                     \
        mul = __riscv_vfmul_vv_f16m##lmul(mul, z, vl);                         \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, z2, one, vl);                    \
        return __riscv_vfmadd_vv_f16m##lmul(y1, mul, add, vl);                 \
    }

REGISTER_RVV_FP16_KERNEL(ACOS_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(acos, _Float16, acos_float16)

// acosh
// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
#define ACOSH_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t acosh_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto sub = __riscv_vfsub_vf_f16m##lmul(v, 1.f16, vl);                  \
        auto add = __riscv_vfadd_vf_f16m##lmul(v, 1.f16, vl);                  \
        auto mul = __riscv_vfmul_vv_f16m##lmul(sub, add, vl);                  \
        auto sqrt = __riscv_vfsqrt_v_f16m##lmul(mul, vl);                      \
        return log_ps_fp16(__riscv_vfadd_vv_f16m##lmul(v, sqrt, vl), vl);      \
    }

REGISTER_RVV_FP16_KERNEL(ACOSH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(acosh, _Float16, acosh_float16)

// asin
#define ASIN_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t asin_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        constexpr auto pc0 = (_Float16)(0x1.921fb6p+0f);                       \
        constexpr auto pc1 = (_Float16)(0x1.55555ep-3);                        \
        constexpr auto pc2 = (_Float16)(0x1.31661ap-4);                        \
        constexpr auto pc3 = (_Float16)(0x1.70d7dcp-5);                        \
        constexpr auto pc4 = (_Float16)(0x1.3af7d8p-5);                        \
        constexpr auto pc5 = (_Float16)(0x1.b059dp-6);                         \
        auto halff = __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl);                  \
        auto one = __riscv_vfmv_v_f_f16m##lmul(1.f16, vl);                     \
        auto minus_two = __riscv_vfmv_v_f_f16m##lmul(-2.f16, vl);              \
        auto pi_over_2f = __riscv_vfmv_v_f_f16m##lmul(pc0, vl);                \
        auto p0 = __riscv_vfmv_v_f_f16m##lmul(pc1, vl);                        \
        auto p1 = __riscv_vfmv_v_f_f16m##lmul(pc2, vl);                        \
        auto p2 = __riscv_vfmv_v_f_f16m##lmul(pc3, vl);                        \
        auto neg_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);   \
        auto x = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        auto mul1 = __riscv_vfmerge_vfm_f16m##lmul(one, -1.f16, neg_mask, vl); \
                                                                               \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto lt_half_mask =                                                    \
            __riscv_vmflt_vv_f16m##lmul##_b##mlen(x, halff, vl);               \
        auto tmp = __riscv_vmv_v_v_f16m##lmul(x, vl);                          \
        auto mul2 = __riscv_vfmerge_vfm_f16m##lmul(minus_two, 1.f16,           \
                                                   lt_half_mask, vl);          \
        tmp = __riscv_vfnmsub_vv_f16m##lmul(tmp, halff, halff, vl);            \
        auto add = __riscv_vfmerge_vfm_f16m##lmul(pi_over_2f, 0.f16,           \
                                                  lt_half_mask, vl);           \
        auto v2 = __riscv_vfmul_vv_f16m##lmul(v, v, vl);                       \
        auto z2 = __riscv_vmerge_vvm_f16m##lmul(tmp, v2, lt_half_mask, vl);    \
        /* asin(|x|) = Q(|x|),        for |x| < 0.5                            \
                = pi / 2 - 2 Q(|x|) , for |x| >= 0.5.  */                      \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(pc4, vl);                        \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(pc5, vl);                        \
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
REGISTER_RVV_UNARY_FP16_OP(asin, _Float16, asin_float16)

// asinh
#define ASINH_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t asinh_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto x = __riscv_vfsgnj_vf_f16##m##lmul(v, 1.f16, vl);                 \
        auto two = __riscv_vfmv_v_f_f16m##lmul(2.f16, vl);                     \
        auto add = __riscv_vfadd_vf_f16m##lmul(x, 1.f16, vl);                  \
        auto sub = __riscv_vfsub_vf_f16m##lmul(x, 1.f16, vl);                  \
        add = __riscv_vfmadd_vv_f16m##lmul(add, sub, two, vl);                 \
        auto sqrt = __riscv_vfsqrt_v_f16m##lmul(add, vl);                      \
        auto ret = log_ps_fp16(__riscv_vfadd_vv_f16m##lmul(x, sqrt, vl), vl);  \
        return __riscv_vfsgnj_vv_f16##m##lmul(ret, v, vl);                     \
    }

REGISTER_RVV_FP16_KERNEL(ASINH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(asinh, _Float16, asinh_float16)

// ceil
#define CEIL_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t ceil_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto vi = __riscv_vfcvt_x_f_v_i16m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f16m##lmul(vi, vl);                      \
        auto mask = __riscv_vmflt_vv_f16m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfadd_vf_f16m##lmul##_m(mask, vf, 1.f16, vl);             \
        return vf;                                                             \
    }

REGISTER_RVV_FP16_KERNEL(CEIL_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(ceil, _Float16, ceil_float16)

// cos
#define COS_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t cos_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        constexpr auto p0 = (_Float16)(0x1.45f306p-2f);                        \
        constexpr auto p1 = (_Float16)(-0x1.555548p-3f);                       \
        constexpr auto p2 = (_Float16)(-0x1.9f42eap-13f);                      \
        constexpr auto p3 = (_Float16)(0x1.921fb6p+1f);                        \
        constexpr auto p4 = (_Float16)(-0x1.777a5cp-24f);                      \
        constexpr auto p5 = (_Float16)(-0x1.ee59dap-49f);                      \
        constexpr auto p6 = (_Float16)(0x1.5b2e76p-19f);                       \
        constexpr auto p7 = (_Float16)(0x1.110df4p-7f);                        \
        auto n = __riscv_vfmv_v_f_f16m##lmul(p0, vl);                          \
        auto halff = __riscv_vfmv_v_f_f16m##lmul(0.5f16, vl);                  \
        auto c0 = __riscv_vfmv_v_f_f16m##lmul(p1, vl);                         \
        auto c2 = __riscv_vfmv_v_f_f16m##lmul(p2, vl);                         \
                                                                               \
        auto r = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        n = __riscv_vfmadd_vv_f16m##lmul(r, n, halff, vl);                     \
        auto ni = __riscv_vfcvt_x_f_v_i16m##lmul(n, vl);                       \
        n = __riscv_vfcvt_f_x_v_f16m##lmul(ni, vl);                            \
        auto parity = __riscv_vand_vx_i16m##lmul(ni, 1, vl);                   \
        auto odd = __riscv_vsll_vx_i16m##lmul(parity, 15, vl);                 \
        n = __riscv_vfsub_vf_f16m##lmul(n, 0.5f16, vl);                        \
                                                                               \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, p3, n, vl);                       \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, p4, n, vl);                       \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, p5, n, vl);                       \
                                                                               \
        auto r2 = __riscv_vfmul_vv_f16m##lmul(r, r, vl);                       \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(p6, vl);                         \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(p7, vl);                         \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r2, c2, vl);                     \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, r2, c0, vl);                     \
        auto r4 = __riscv_vfmul_vv_f16m##lmul(r2, r2, vl);                     \
        auto r3 = __riscv_vfmul_vv_f16m##lmul(r2, r, vl);                      \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r4, y2, vl);                     \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r3, r, vl);                      \
                                                                               \
        auto tmp = __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(y1);         \
        tmp = __riscv_vxor_vv_i16m##lmul(tmp, odd, vl);                        \
        return __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(tmp);            \
    }

REGISTER_RVV_FP16_KERNEL(COS_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(cos, _Float16, cos_float16)

// cosh
#define COSH_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t cosh_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps_fp16(v, vl);                                           \
        auto b = __riscv_vfrdiv_vf_f16m##lmul(a, 1.f16, vl);                   \
        auto sum = __riscv_vfadd_vv_f16m##lmul(a, b, vl);                      \
        return __riscv_vfmul_vf_f16m##lmul(sum, 0.5f16, vl);                   \
    }

REGISTER_RVV_FP16_KERNEL(COSH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(cosh, _Float16, cosh_float16)

// floor
#define FLOOR_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t floor_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto vi = __riscv_vfcvt_x_f_v_i16m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f16m##lmul(vi, vl);                      \
        auto mask = __riscv_vmfgt_vv_f16m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfsub_vf_f16m##lmul##_m(mask, vf, 1.f16, vl);             \
        return vf;                                                             \
    }

REGISTER_RVV_FP16_KERNEL(FLOOR_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(floor, _Float16, floor_float16)

// neg
#define NEG_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t neg_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return __riscv_vfneg_v_f16m##lmul(v, vl);                              \
    }

REGISTER_RVV_FP16_KERNEL(NEG_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(neg, _Float16, neg_float16)

// rsqrt
#define RSQRT_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t rsqrt_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto one_point_five = __riscv_vfmv_v_f_f16m##lmul(1.5f16, vl);         \
                                                                               \
        auto ux = __riscv_vreinterpret_v_f16m##lmul##_u16m##lmul(v);           \
        ux = __riscv_vsrl_vx_u16m##lmul(ux, 1, vl);                            \
        ux = __riscv_vrsub_vx_u16m##lmul(                                      \
            ux, static_cast<uint16_t>(0x5f375a86), vl);                        \
        auto y = __riscv_vreinterpret_v_u16m##lmul##_f16m##lmul(ux);           \
                                                                               \
        auto y2 = __riscv_vfmul_vv_f16m##lmul(y, y, vl);                       \
        auto x = __riscv_vfmul_vf_f16m##lmul(v, -0.5f16, vl);                  \
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
REGISTER_RVV_UNARY_FP16_OP(rsqrt, _Float16, rsqrt_float16)

// round
#define ROUND_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t round_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        return __riscv_vfcvt_f_x_v_f16m##lmul(                                 \
            __riscv_vfcvt_x_f_v_i16m##lmul(v, vl), vl);                        \
    }

REGISTER_RVV_FP16_KERNEL(ROUND_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(round, _Float16, round_float16)

// sign
#define SIGN_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t sign_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto ret = __riscv_vfmv_v_f_f16m##lmul(0.f16, vl);                     \
        auto gt_mask = __riscv_vmfgt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);    \
        ret = __riscv_vfmerge_vfm_f16m##lmul(ret, 1.f16, gt_mask, vl);         \
        auto lt_mask = __riscv_vmflt_vf_f16m##lmul##_b##mlen(v, 0.f16, vl);    \
        return __riscv_vfmerge_vfm_f16m##lmul(ret, -1.f16, lt_mask, vl);       \
    }

REGISTER_RVV_FP16_KERNEL(SIGN_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(sign, _Float16, sign_float16)

// sin
#define SIN_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t sin_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        constexpr auto pc0 = (_Float16)(-0x1.555548p-3f);                      \
        constexpr auto pc1 = (_Float16)(-0x1.9f42eap-13f);                     \
        constexpr auto pc2 = (_Float16)(0x1.45f306p-2f);                       \
        constexpr auto pc3 = (_Float16)(0x1.921fb6p+1f);                       \
        constexpr auto pc4 = (_Float16)(-0x1.777a5cp-24f);                     \
        constexpr auto pc5 = (_Float16)(-0x1.ee59dap-49f);                     \
        constexpr auto pc6 = (_Float16)(0x1.5b2e76p-19f);                      \
        constexpr auto pc7 = (_Float16)(0x1.110df4p-7f);                       \
        auto c0 = __riscv_vfmv_v_f_f16m##lmul(pc0, vl);                        \
        auto c2 = __riscv_vfmv_v_f_f16m##lmul(pc1, vl);                        \
                                                                               \
        /* n = rint(|x|/pi) */                                                 \
        auto r = __riscv_vfabs_v_f16m##lmul(v, vl);                            \
        auto n = __riscv_vfmul_vf_f16m##lmul(r, pc2, vl);                      \
        auto sign = __riscv_vxor_vv_i16m##lmul(                                \
            __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(v),                 \
            __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(r), vl);            \
        auto ni = __riscv_vfcvt_x_f_v_i16m##lmul(n, vl);                       \
        n = __riscv_vfcvt_f_x_v_f16m##lmul(ni, vl);                            \
        auto odd = __riscv_vand_vx_i16m##lmul(ni, 1, vl);                      \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, pc3, n, vl);                      \
        odd = __riscv_vsll_vx_i16m##lmul(odd, 15, vl);                         \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, pc4, n, vl);                      \
        r = __riscv_vfnmsac_vf_f16m##lmul(r, pc5, n, vl);                      \
                                                                               \
        /* y = sin(r).  */                                                     \
        auto r2 = __riscv_vfmul_vv_f16m##lmul(r, r, vl);                       \
        auto y1 = __riscv_vfmv_v_f_f16m##lmul(pc6, vl);                        \
        auto y2 = __riscv_vfmv_v_f_f16m##lmul(pc7, vl);                        \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r2, c2, vl);                     \
        y2 = __riscv_vfmadd_vv_f16m##lmul(y2, r2, c0, vl);                     \
        auto r4 = __riscv_vfmul_vv_f16m##lmul(r2, r2, vl);                     \
        auto r3 = __riscv_vfmul_vv_f16m##lmul(r2, r, vl);                      \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r4, y2, vl);                     \
        sign = __riscv_vxor_vv_i16m##lmul(sign, odd, vl);                      \
        y1 = __riscv_vfmadd_vv_f16m##lmul(y1, r3, r, vl);                      \
        auto tmp = __riscv_vreinterpret_v_f16m##lmul##_i16m##lmul(y1);         \
        tmp = __riscv_vxor_vv_i16m##lmul(tmp, sign, vl);                       \
        return __riscv_vreinterpret_v_i16m##lmul##_f16m##lmul(tmp);            \
    }

REGISTER_RVV_FP16_KERNEL(SIN_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(sin, _Float16, sin_float16)

// sinh
// sinh(v) = (exp(v) - exp(-v)) / 2
#define SINH_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t sinh_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps_fp16(v, vl);                                           \
        auto b = __riscv_vfrdiv_vf_f16m##lmul(a, 1.f16, vl);                   \
        return __riscv_vfmul_vf_f16m##lmul(                                    \
            __riscv_vfsub_vv_f16m##lmul(a, b, vl), 0.5f16, vl);                \
    }

REGISTER_RVV_FP16_KERNEL(SINH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(sinh, _Float16, sinh_float16)

// sqrt
#define SQRT_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t sqrt_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        return __riscv_vfsqrt_v_f16m##lmul(v, vl);                             \
    }

REGISTER_RVV_FP16_KERNEL(SQRT_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(sqrt, _Float16, sqrt_float16)


// square
#define SQUARE_FLOAT16(lmul, mlen)                                             \
    inline vfloat16m##lmul##_t square_float16(const vfloat16m##lmul##_t &v,    \
                                              const size_t vl) {               \
        return __riscv_vfmul_vv_f16m##lmul(v, v, vl);                          \
    }

REGISTER_RVV_FP16_KERNEL(SQUARE_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(square, _Float16, square_float16)

// exp
#define EXP_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t exp_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return exp_ps_fp16(v, vl);                                             \
    }

REGISTER_RVV_FP16_KERNEL(EXP_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(exp, _Float16, exp_float16)

// log
#define LOG_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t log_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return log_ps_fp16(v, vl);                                             \
    }

REGISTER_RVV_FP16_KERNEL(LOG_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(log, _Float16, log_float16)

// tanh
#define TANH_FLOAT16(lmul, mlen)                                               \
    inline vfloat16m##lmul##_t tanh_float16(const vfloat16m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        return tanh_ps_fp16(v, vl);                                            \
    }

REGISTER_RVV_FP16_KERNEL(TANH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(tanh, _Float16, tanh_float16)

// swish
// swish(v) = v / (exp(-v) + 1)
#define SWISH_FLOAT16(lmul, mlen)                                              \
    inline vfloat16m##lmul##_t swish_float16(const vfloat16m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto tmp = exp_ps_fp16(__riscv_vfneg_v_f16m##lmul(v, vl), vl);         \
        return __riscv_vfdiv_vv_f16m##lmul(                                    \
            v, __riscv_vfadd_vf_f16m##lmul(tmp, 1.f16, vl), vl);               \
    }

REGISTER_RVV_FP16_KERNEL(SWISH_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(swish, _Float16, swish_float16)

// register swishb kernel
// swishb(v) = v / (exp(-v*beta) + 1)
#define SWISHB_FLOAT16(lmul, mlen)                                             \
    inline vfloat16m##lmul##_t swishb_float16(                                 \
        const vfloat16m##lmul##_t &v, _Float16 beta, const size_t vl) {        \
        auto tmp = __riscv_vfmul_vf_f16m##lmul(v, -beta, vl);                  \
        tmp = exp_ps_fp16(tmp, vl);                                            \
        tmp = __riscv_vfadd_vf_f16m##lmul(tmp, 1.f16, vl);                     \
        return __riscv_vfdiv_vv_f16m##lmul(v, tmp, vl);                        \
    }

REGISTER_RVV_FP16_KERNEL(SWISHB_FLOAT16)

// register swishb op
#define RVV_SWISHB_FP16_OP(dtype, vl, kernel)                                  \
    template <> struct swishb<ntt::vector<dtype, vl>, dtype> {                 \
        ntt::vector<dtype, vl> operator()(const ntt::vector<dtype, vl> &v,     \
                                          const dtype &beta) const noexcept {  \
            return kernel(v, beta, vl);                                        \
        }                                                                      \
    };

#define REGISTER_RVV_SWISHB_FP16_OP(dtype, kernel)                             \
    RVV_SWISHB_FP16_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)         \
    RVV_SWISHB_FP16_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)         \
    RVV_SWISHB_FP16_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)         \
    RVV_SWISHB_FP16_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_SWISHB_FP16_OP(_Float16, swishb_float16)

// erf
#define ERF_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t erf_float16(const vfloat16m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return erf_ps_fp16(v, vl);                                             \
    }

REGISTER_RVV_FP16_KERNEL(ERF_FLOAT16)
REGISTER_RVV_UNARY_FP16_OP(erf, _Float16, erf_float16)

// binary
#define RVV_BINARY_fp16_OP(op, dtype, vl, kernel)                              \
    template <> struct op<ntt::vector<dtype, vl>, ntt::vector<dtype, vl>> {    \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v1,                           \
                   const ntt::vector<dtype, vl> &v2) const noexcept {          \
            return kernel(v1, v2, vl);                                         \
        }                                                                      \
    };                                                                         \
    template <> struct op<ntt::vector<dtype, vl>, dtype> {                     \
        ntt::vector<dtype, vl> operator()(const ntt::vector<dtype, vl> &v,     \
                                          const dtype &s) const noexcept {     \
            return kernel(v, s, vl);                                           \
        }                                                                      \
    };                                                                         \
    template <> struct op<dtype, ntt::vector<dtype, vl>> {                     \
        ntt::vector<dtype, vl>                                                 \
        operator()(const dtype &s,                                             \
                   const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(s, v, vl);                                           \
        };                                                                     \
    };

// binary op
#define REGISTER_RVV_BINARY_FP16_OP(op, dtype, kernel)                         \
    RVV_BINARY_fp16_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)     \
    RVV_BINARY_fp16_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)     \
    RVV_BINARY_fp16_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)     \
    RVV_BINARY_fp16_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)     

// add
#define ADD_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t add_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfadd_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t add_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfadd_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t add_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfadd_vf_f16m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_FP16_KERNEL(ADD_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(add, _Float16, add_float16)

// sub
#define SUB_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t sub_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfsub_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t sub_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfsub_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t sub_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfrsub_vf_f16m##lmul(v, s, vl);                         \
    }

REGISTER_RVV_FP16_KERNEL(SUB_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(sub, _Float16, sub_float16)

// mul
#define MUL_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t mul_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmul_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t mul_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfmul_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t mul_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfmul_vf_f16m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_FP16_KERNEL(MUL_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(mul, _Float16, mul_float16)

// div
#define DIV_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t div_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfdiv_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t div_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfdiv_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t div_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfrdiv_vf_f16m##lmul(v, s, vl);                         \
    }

REGISTER_RVV_FP16_KERNEL(DIV_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(div, _Float16, div_float16)

// pow
#define POW_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t pow_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return pow_ps_fp16(v1, v2, vl);                                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t pow_float16(                                    \
        const vfloat16m##lmul##_t &v1, const _Float16 &s, const size_t vl) {   \
        auto v2 = __riscv_vfmv_v_f_f16m##lmul(s, vl);                          \
        return pow_ps_fp16(v1, v2, vl);                                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t pow_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v2, const size_t vl) {   \
        auto v1 = __riscv_vfmv_v_f_f16m##lmul(s, vl);                          \
        return pow_ps_fp16(v1, v2, vl);                                        \
    }

REGISTER_RVV_FP16_KERNEL(POW_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(pow, _Float16, pow_float16)

// mod
#define MOD_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t mod_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        auto quotient = __riscv_vfcvt_f_x_v_f16m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i16m##lmul(                                \
                __riscv_vfdiv_vv_f16m##lmul(v1, v2, vl), vl),                  \
            vl);                                                               \
        return __riscv_vfnmsub_vv_f16m##lmul(quotient, v2, v1, vl);            \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t mod_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        auto quotient = __riscv_vfcvt_f_x_v_f16m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i16m##lmul(                                \
                __riscv_vfdiv_vf_f16m##lmul(v, s, vl), vl),                    \
            vl);                                                               \
        return __riscv_vfnmsub_vf_f16m##lmul(quotient, s, v, vl);              \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t mod_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v2, const size_t vl) {   \
        auto v1 = __riscv_vfmv_v_f_f16m##lmul(s, vl);                          \
        auto quotient = __riscv_vfcvt_f_x_v_f16m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i16m##lmul(                                \
                __riscv_vfrdiv_vf_f16m##lmul(v2, s, vl), vl),                  \
            vl);                                                               \
        return __riscv_vfnmsub_vv_f16m##lmul(quotient, v2, v1, vl);            \
    }

REGISTER_RVV_FP16_KERNEL(MOD_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(mod, _Float16, mod_float16)

#define MIN_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t min_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmin_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t min_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfmin_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t min_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfmin_vf_f16m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_FP16_KERNEL(MIN_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(min, _Float16, min_float16)

#define MAX_FLOAT16(lmul, mlen)                                                \
    inline vfloat16m##lmul##_t max_float16(const vfloat16m##lmul##_t &v1,      \
                                           const vfloat16m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmax_vv_f16m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t max_float16(                                    \
        const vfloat16m##lmul##_t &v, const _Float16 &s, const size_t vl) {    \
        return __riscv_vfmax_vf_f16m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat16m##lmul##_t max_float16(                                    \
        const _Float16 &s, const vfloat16m##lmul##_t &v, const size_t vl) {    \
        return __riscv_vfmax_vf_f16m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_FP16_KERNEL(MAX_FLOAT16)
REGISTER_RVV_BINARY_FP16_OP(max, _Float16, max_float16)

// floor_mod
#define FLOOR_MOD_INT16(lmul, mlen)                                            \
    inline vint16m##lmul##_t floor_mod_int16(const vint16m##lmul##_t &v1,      \
                                             const vint16m##lmul##_t &v2,      \
                                             const size_t vl) {                \
        auto remainder = __riscv_vrem_vv_i16m##lmul(v1, v2, vl);               \
        auto tmp = __riscv_vxor_vv_i16m##lmul(v1, v2, vl);                     \
        auto mask1 = __riscv_vmsne_vx_i16m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i16m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vv_i16m##lmul##_m(mask1, remainder, v2, vl);  \
        return remainder;                                                      \
    }                                                                          \
                                                                               \
    inline vint16m##lmul##_t floor_mod_int16(                                  \
        const vint16m##lmul##_t &v1, const int16_t &s, const size_t vl) {      \
        auto remainder = __riscv_vrem_vx_i16m##lmul(v1, s, vl);                \
        auto tmp = __riscv_vxor_vx_i16m##lmul(v1, s, vl);                      \
        auto mask1 = __riscv_vmsne_vx_i16m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i16m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vx_i16m##lmul##_m(mask1, remainder, s, vl);   \
        return remainder;                                                      \
    }                                                                          \
                                                                               \
    inline vint16m##lmul##_t floor_mod_int16(                                  \
        const int16_t &s, const vint16m##lmul##_t &v2, const size_t vl) {      \
        auto v1 = __riscv_vmv_v_x_i16m##lmul(s, vl);                           \
        auto remainder = __riscv_vrem_vv_i16m##lmul(v1, v2, vl);               \
        auto tmp = __riscv_vxor_vv_i16m##lmul(v1, v2, vl);                     \
        auto mask1 = __riscv_vmsne_vx_i16m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i16m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vv_i16m##lmul##_m(mask1, remainder, v2, vl);  \
        return remainder;                                                      \
    }

REGISTER_RVV_FP16_KERNEL(FLOOR_MOD_INT16)
REGISTER_RVV_BINARY_FP16_OP(floor_mod, int16_t, floor_mod_int16)

} // namespace nncase::ntt::ops
