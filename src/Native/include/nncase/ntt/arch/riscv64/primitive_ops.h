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
#include "../../primitive_ops.h"
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/vector.h"
#include "rvv_mathfun.h"

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace nncase::ntt::ops {

#ifdef __riscv_vector

#ifndef REGISTER_RVV_KERNEL
#define REGISTER_RVV_KERNEL(kernel)                                            \
    kernel(1, 32) kernel(2, 16) kernel(4, 8) kernel(8, 4)
#endif

#ifndef REGISTER_RVV_KERNEL_1_4
#define REGISTER_RVV_KERNEL_1_4(kernel)                                        \
    kernel(f8, f2, 64) kernel(f4, 1, 32) kernel(f2, 2, 16) kernel(1, 4, 8)     \
        kernel(2, 8, 4)
#endif

#ifndef REGISTER_RVV_KERNEL_4_1
#define REGISTER_RVV_KERNEL_4_1(kernel)                                        \
    kernel(1, f2, f4, 32) kernel(2, 1, f2, 16) kernel(4, 2, 1, 8)              \
        kernel(8, 4, 2, 4)
#endif

template <>
struct store<ntt::vector<float, NTT_VLEN / 32>,
             ntt::vector<float, NTT_VLEN / 32>> {
    void operator()(ntt::vector<float, NTT_VLEN / 32> &dest,
                    const ntt::vector<float, NTT_VLEN / 32> &v) const noexcept {
        __riscv_vse32_v_f32m1((float *)&dest, v, NTT_VLEN / 32);
    }
};

#define RVV_UNARY_OP(op, dtype, vl, kernel)                                    \
    template <> struct op<ntt::vector<dtype, vl>> {                            \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

// unary with float
#define REGISTER_RVV_UNARY_OP(OP, dtype, kernel)                               \
    RVV_UNARY_OP(OP, float, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)           \
    RVV_UNARY_OP(OP, float, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)           \
    RVV_UNARY_OP(OP, float, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)           \
    RVV_UNARY_OP(OP, float, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

// abs
#define ABS_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t abs_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return __riscv_vfabs_v_f32m##lmul(v, vl);                              \
    }

REGISTER_RVV_KERNEL(ABS_FLOAT32)
REGISTER_RVV_UNARY_OP(abs, float, abs_float32)

// acos
#if 0
// max_ulp_error = 789 on c908
// porting from https://developer.download.nvidia.cn/cg/acos.html
#define ACOS_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t acos_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto c2 = __riscv_vfmv_v_f_f32m##lmul(0.0742610f, vl);                 \
        auto c3 = __riscv_vfmv_v_f_f32m##lmul(-0.2121144f, vl);                \
        auto c4 = __riscv_vfmv_v_f_f32m##lmul(1.5707288f, vl);                 \
        auto c5 = __riscv_vfmv_v_f_f32m##lmul(3.14159265358979f, vl);          \
        auto x = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        auto mask = __riscv_vmflt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);         \
        auto sroot =                                                           \
            __riscv_vfsqrt_v_f32m##lmul(vfrsub_vf_f32m##lmul(x, 1.f, vl), vl); \
        auto ret = __riscv_vmv_v_v_f32m##lmul(x, vl);                          \
        ret = __riscv_vfmadd_vf_f32m##lmul(ret, -0.0187293f, c2, vl);          \
        ret = __riscv_vfmadd_vv_f32m##lmul(ret, x, c3, vl);                    \
        ret = __riscv_vfmadd_vv_f32m##lmul(ret, x, c4, vl);                    \
        ret = __riscv_vfmul_vv_f32m##lmul(ret, sroot, vl);                     \
        return __riscv_vfmadd_vf_f32m##lmul##_m(mask, ret, -1.f, c5, vl);      \
    }
#else
// from glibc 2.40: sysdeps/aarch64/fpu/acosf_advsimd.c
#define ACOS_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t acos_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto zero = __riscv_vfmv_v_f_f32m##lmul(0.f, vl);                      \
        auto half = __riscv_vfmv_v_f_f32m##lmul(0.5f, vl);                     \
        auto one = __riscv_vfmv_v_f_f32m##lmul(1.f, vl);                       \
        auto two = __riscv_vfmv_v_f_f32m##lmul(2.f, vl);                       \
        auto minus_one = __riscv_vfmv_v_f_f32m##lmul(-1.f, vl);                \
        auto p0 = __riscv_vfmv_v_f_f32m##lmul(0x1.55555ep-3, vl);              \
        auto p1 = __riscv_vfmv_v_f_f32m##lmul(0x1.33261ap-4, vl);              \
        auto p2 = __riscv_vfmv_v_f_f32m##lmul(0x1.70d7dcp-5, vl);              \
        auto neg_mask = __riscv_vmflt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);     \
        auto x = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        auto off = __riscv_vfmerge_vfm_f32m##lmul(zero, 0x1.921fb6p+1f,        \
                                                  neg_mask, vl);               \
        auto mul1 = __riscv_vfmerge_vfm_f32m##lmul(two, -2.f, neg_mask, vl);   \
        auto mul2 =                                                            \
            __riscv_vfmerge_vfm_f32m##lmul(minus_one, 1.f, neg_mask, vl);      \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto le_half_mask =                                                    \
            __riscv_vmfle_vv_f32m##lmul##_b##mlen(x, half, vl);                \
        auto tmp = __riscv_vmv_v_v_f32m##lmul(x, vl);                          \
        auto mul =                                                             \
            __riscv_vmerge_vvm_f32m##lmul(mul1, mul2, le_half_mask, vl);       \
        tmp = __riscv_vfnmsub_vv_f32m##lmul(tmp, half, half, vl);              \
        auto v2 = __riscv_vfmul_vv_f32m##lmul(v, v, vl);                       \
        auto add = __riscv_vfmerge_vfm_f32m##lmul(off, 0x1.921fb6p+0f,         \
                                                  le_half_mask, vl);           \
        auto z2 = __riscv_vmerge_vvm_f32m##lmul(tmp, v2, le_half_mask, vl);    \
        /* acos(|x|) = pi/2 - sign(x) * Q(|x|), for  |x| < 0.5                 \
                = 2 Q(|x|)               , for  0.5 < x < 1.0                  \
                = pi - 2 Q(|x|)          , for -1.0 < x < -0.5.  */            \
        auto y1 = __riscv_vfmv_v_f_f32m##lmul(0x1.3af7d8p-5, vl);              \
        auto y2 = __riscv_vfmv_v_f_f32m##lmul(0x1.b059dp-6, vl);               \
        tmp = __riscv_vfsqrt_v_f32m##lmul(z2, vl);                             \
        auto z4 = __riscv_vfmul_vv_f32m##lmul(z2, z2, vl);                     \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z4, p2, vl);                     \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, z4, p1, vl);                     \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z4, p0, vl);                     \
        auto z = __riscv_vmerge_vvm_f32m##lmul(tmp, x, le_half_mask, vl);      \
        y1 = __riscv_vfmacc_vv_f32m##lmul(y1, y2, z2, vl);                     \
        mul = __riscv_vfmul_vv_f32m##lmul(mul, z, vl);                         \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z2, one, vl);                    \
        return __riscv_vfmadd_vv_f32m##lmul(y1, mul, add, vl);                 \
    }
#endif

REGISTER_RVV_KERNEL(ACOS_FLOAT32)
REGISTER_RVV_UNARY_OP(acos, float, acos_float32)

// acosh
// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
#if 1
#define ACOSH_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t acosh_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto sub = __riscv_vfsub_vf_f32m##lmul(v, 1.f, vl);                    \
        auto add = __riscv_vfadd_vf_f32m##lmul(v, 1.f, vl);                    \
        auto mul = __riscv_vfmul_vv_f32m##lmul(sub, add, vl);                  \
        auto sqrt = __riscv_vfsqrt_v_f32m##lmul(mul, vl);                      \
        return log_ps(__riscv_vfadd_vv_f32m##lmul(v, sqrt, vl), vl);           \
    }
#else
#define ACOSH_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t acosh_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto minus_one = __riscv_vfmv_v_f_f32m##lmul(-1.f, vl);                \
        auto minus_half = __riscv_vfmv_v_f_f32m##lmul(-0.5f, vl);              \
        auto poly_1 = __riscv_vfmv_v_f_f32m##lmul(-0x1.000038p-2f, vl);        \
        auto poly_3 = __riscv_vfmv_v_f_f32m##lmul(-0x1.54ef78p-3f, vl);        \
        auto poly_5 = __riscv_vfmv_v_f_f32m##lmul(-0x1.0da91p-3f, vl);         \
        auto xm1 = __riscv_vfsub_vf_f32m##lmul(v, 1.f, vl);                    \
        auto u = __riscv_vfmul_vv_f32m##lmul(                                  \
            xm1, __riscv_vfadd_vf_f32m##lmul(v, 1.f, vl), vl);                 \
        auto x =                                                               \
            __riscv_vfadd_vv_f32m##lmul(xm1, vfsqrt_v_f32m##lmul(u, vl), vl);  \
        auto m = __riscv_vfadd_vf_f32m##lmul(x, 1.f, vl);                      \
        auto ks = __riscv_vsub_vx_i32m##lmul(                                  \
            __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(m), 0x3f400000,     \
            vl);                                                               \
        auto k = __riscv_vand_vx_i32m##lmul(ks, 0xff800000, vl);               \
        auto ku = __riscv_vreinterpret_v_i32m##lmul##_u32m##lmul(k);           \
        auto s = __riscv_vreinterpret_v_u32m##lmul##_f32m##lmul(               \
            __riscv_vrsub_vx_u32m##lmul(ku, 0x40800000, vl));                  \
        auto m_scale = __riscv_vreinterpret_v_u32m##lmul##_f32m##lmul(         \
            __riscv_vsub_vv_u32m##lmul(                                        \
                __riscv_vreinterpret_v_f32m##lmul##_u32m##lmul(x), ku, vl));   \
        m_scale = __riscv_vfadd_vv_f32m##lmul(                                 \
            m_scale, __riscv_vfmadd_vf_f32m##lmul(s, 0.25f, minus_one, vl),    \
            vl);                                                               \
        /* eval_poly*/                                                         \
        auto q = __riscv_vmv_v_v_f32m##lmul(m_scale, vl);                      \
        q = __riscv_vfmadd_vf_f32m##lmul(q, 0x1.5555aap-2f, minus_half, vl);   \
        auto m2 = __riscv_vfmul_vv_f32m##lmul(m_scale, m_scale, vl);           \
        q = __riscv_vfmadd_vv_f32m##lmul(q, m2, m_scale, vl);                  \
        /*float32x4_t p = v_pw_horner_6_f32 (m, m2, c + 1);*/                  \
        auto p01 = __riscv_vmv_v_v_f32m##lmul(m_scale, vl);                    \
        auto p23 = __riscv_vmv_v_v_f32m##lmul(m_scale, vl);                    \
        auto p = __riscv_vmv_v_v_f32m##lmul(m2, vl);                           \
        p01 = __riscv_vfmadd_vf_f32m##lmul(p01, 0x1.28a1f4p-3f, poly_3, vl);   \
        p23 = __riscv_vfmadd_vf_f32m##lmul(p23, 0x1.abcb6p-4f, poly_5, vl);    \
        p = __riscv_vfmadd_vf_f32m##lmul(p, -0x1.6f0d5ep-5f, p23, vl);         \
        p = __riscv_vfmadd_vv_f32m##lmul(p, m2, p01, vl);                      \
        m_scale =                                                              \
            __riscv_vfmadd_vf_f32m##lmul(m_scale, 0x1.99675cp-3f, poly_1, vl); \
        p = __riscv_vfmadd_vv_f32m##lmul(p, m2, m_scale, vl);                  \
        p = __riscv_vfmul_vv_f32m##lmul(p, m2, vl);                            \
        p = __riscv_vfmadd_vv_f32m##lmul(p, m2, q, vl);                        \
        auto scale_back = __riscv_vfmul_vf_f32m##lmul(                         \
            __riscv_vreinterpret_v_u32m##lmul##_f32m##lmul(ku), 0x1.0p-23f,    \
            vl);                                                               \
        return __riscv_vfmadd_vf_f32m##lmul(scale_back, 0x1.62e43p-1f, p, vl); \
    }
#endif

REGISTER_RVV_KERNEL(ACOSH_FLOAT32)
REGISTER_RVV_UNARY_OP(acosh, float, acosh_float32)

// asin
#if 0
// porting from https://developer.download.nvidia.cn/cg/asin.html
#define ASIN_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t asin_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto c2 = __riscv_vfmv_v_f_f32m##lmul(0.0742610f, vl);                 \
        auto c3 = __riscv_vfmv_v_f_f32m##lmul(-0.2121144f, vl);                \
        auto c4 = __riscv_vfmv_v_f_f32m##lmul(1.5707288f, vl);                 \
        auto c5 = __riscv_vfmv_v_f_f32m##lmul(3.14159265358979f * 0.5f, vl);   \
        auto x = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        auto mask = __riscv_vmflt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);         \
        auto sroot = __riscv_vfsqrt_v_f32m##lmul(                              \
            __riscv_vfrsub_vf_f32m##lmul(x, 1.f, vl), vl);                     \
        auto ret = __riscv_vmv_v_v_f32m##lmul(x, vl);                          \
        ret = __riscv_vfmadd_vf_f32m##lmul(ret, -0.0187293f, c2, vl);          \
        ret = __riscv_vfmadd_vv_f32m##lmul(ret, x, c3, vl);                    \
        ret = __riscv_vfmadd_vv_f32m##lmul(ret, x, c4, vl);                    \
        ret = __riscv_vfnmsub_vv_f32m##lmul(ret, sroot, c5, vl);               \
        return __riscv_vfneg_v_f32m##lmul##_m(mask, ret, ret, vl);             \
    }
#else
// from glibc 2.40: sysdeps/aarch64/fpu/asinf_advsimd.c
#define ASIN_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t asin_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto half = __riscv_vfmv_v_f_f32m##lmul(0.5f, vl);                     \
        auto one = __riscv_vfmv_v_f_f32m##lmul(1.f, vl);                       \
        auto minus_two = __riscv_vfmv_v_f_f32m##lmul(-2.f, vl);                \
        auto pi_over_2f = __riscv_vfmv_v_f_f32m##lmul(0x1.921fb6p+0f, vl);     \
        auto p0 = __riscv_vfmv_v_f_f32m##lmul(0x1.55555ep-3, vl);              \
        auto p1 = __riscv_vfmv_v_f_f32m##lmul(0x1.33261ap-4, vl);              \
        auto p2 = __riscv_vfmv_v_f_f32m##lmul(0x1.70d7dcp-5, vl);              \
        auto neg_mask = __riscv_vmflt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);     \
        auto x = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        auto mul1 = __riscv_vfmerge_vfm_f32m##lmul(one, -1.f, neg_mask, vl);   \
                                                                               \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto lt_half_mask =                                                    \
            __riscv_vmflt_vv_f32m##lmul##_b##mlen(x, half, vl);                \
        auto tmp = __riscv_vmv_v_v_f32m##lmul(x, vl);                          \
        auto mul2 =                                                            \
            __riscv_vfmerge_vfm_f32m##lmul(minus_two, 1.f, lt_half_mask, vl);  \
        tmp = __riscv_vfnmsub_vv_f32m##lmul(tmp, half, half, vl);              \
        auto add =                                                             \
            __riscv_vfmerge_vfm_f32m##lmul(pi_over_2f, 0.f, lt_half_mask, vl); \
        auto v2 = __riscv_vfmul_vv_f32m##lmul(v, v, vl);                       \
        auto z2 = __riscv_vmerge_vvm_f32m##lmul(tmp, v2, lt_half_mask, vl);    \
        /* asin(|x|) = Q(|x|),        for |x| < 0.5                            \
                = pi / 2 - 2 Q(|x|) , for |x| >= 0.5.  */                      \
        auto y1 = __riscv_vfmv_v_f_f32m##lmul(0x1.3af7d8p-5, vl);              \
        auto y2 = __riscv_vfmv_v_f_f32m##lmul(0x1.b059dp-6, vl);               \
        auto z4 = __riscv_vfmul_vv_f32m##lmul(z2, z2, vl);                     \
        tmp = __riscv_vfsqrt_v_f32m##lmul(z2, vl);                             \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z4, p2, vl);                     \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, z4, p1, vl);                     \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z4, p0, vl);                     \
        auto z = __riscv_vmerge_vvm_f32m##lmul(tmp, x, lt_half_mask, vl);      \
        y1 = __riscv_vfmacc_vv_f32m##lmul(y1, y2, z2, vl);                     \
        z2 = __riscv_vfmul_vv_f32m##lmul(z2, z, vl);                           \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, z2, z, vl);                      \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, mul2, add, vl);                  \
        return __riscv_vfmul_vv_f32m##lmul(y1, mul1, vl);                      \
    }
#endif

REGISTER_RVV_KERNEL(ASIN_FLOAT32)
REGISTER_RVV_UNARY_OP(asin, float, asin_float32)

// asinh
// asinh(v) = ln(v + sqrt(v^2 + 1)), -inf < x < +inf
#if 0
#define ASINH_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t asinh_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto sum = __riscv_vfmv_v_f_f32m##lmul(1.f, vl);                       \
        auto x = __riscv_vfsgnj_vf_f32##m##lmul(v, 1.f, vl);                   \
        sum = __riscv_vfmacc_vv_f32m##lmul(sum, v, v, vl);                     \
        auto sqrt = __riscv_vfrec7_v_f32m##lmul(                               \
            __riscv_vfrsqrt7_v_f32m##lmul(sum, vl), vl);                       \
        auto ret = log_ps(__riscv_vfadd_vv_f32m##lmul(x, sqrt, vl), vl);       \
        return __riscv_vfsgnj_vv_f32##m##lmul(ret, v, vl);                     \
    }
#else
#define ASINH_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t asinh_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto x = __riscv_vfsgnj_vf_f32##m##lmul(v, 1.f, vl);                   \
        auto two = __riscv_vfmv_v_f_f32m##lmul(2.f, vl);                       \
        auto add = __riscv_vfadd_vf_f32m##lmul(x, 1.f, vl);                    \
        auto sub = __riscv_vfsub_vf_f32m##lmul(x, 1.f, vl);                    \
        add = __riscv_vfmadd_vv_f32m##lmul(add, sub, two, vl);                 \
        auto sqrt = __riscv_vfsqrt_v_f32m##lmul(add, vl);                      \
        auto ret = log_ps(__riscv_vfadd_vv_f32m##lmul(x, sqrt, vl), vl);       \
        return __riscv_vfsgnj_vv_f32##m##lmul(ret, v, vl);                     \
    }
#endif

REGISTER_RVV_KERNEL(ASINH_FLOAT32)
REGISTER_RVV_UNARY_OP(asinh, float, asinh_float32)

// ceil
#define CEIL_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t ceil_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto vi = __riscv_vfcvt_x_f_v_i32m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f32m##lmul(vi, vl);                      \
        auto mask = __riscv_vmflt_vv_f32m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfadd_vf_f32m##lmul##_m(mask, vf, 1.f, vl);               \
        return vf;                                                             \
    }

REGISTER_RVV_KERNEL(CEIL_FLOAT32)
REGISTER_RVV_UNARY_OP(ceil, float, ceil_float32)

// cos
// from glibc 2.40: sysdeps/aarch64/fpu/cosf_advsimd.c
#define COS_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t cos_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        auto n = __riscv_vfmv_v_f_f32m##lmul(0x1.45f306p-2f, vl);              \
        auto half = __riscv_vfmv_v_f_f32m##lmul(0.5f, vl);                     \
        auto c0 = __riscv_vfmv_v_f_f32m##lmul(-0x1.555548p-3f, vl);            \
        auto c2 = __riscv_vfmv_v_f_f32m##lmul(-0x1.9f42eap-13f, vl);           \
                                                                               \
        /*  n = rint((|x|+pi/2)/pi) - 0.5. */                                  \
        auto r = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        n = __riscv_vfmadd_vv_f32m##lmul(n, r, half, vl);                      \
        auto ni = __riscv_vfcvt_x_f_v_i32m##lmul(n, vl);                       \
        n = __riscv_vfcvt_f_x_v_f32m##lmul(ni, vl);                            \
        auto odd = __riscv_vadd_vx_i32m##lmul(ni, 0x1.8p+23, vl);              \
        n = __riscv_vfsub_vf_f32m##lmul(n, 0.5f, vl);                          \
        odd = __riscv_vsll_vx_i32##m##lmul(odd, 31, vl);                       \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, 0x1.921fb6p+1f, n, vl);           \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.777a5cp-24f, n, vl);         \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.ee59dap-49f, n, vl);         \
                                                                               \
        /* y = sin(r).  */                                                     \
        auto r2 = __riscv_vfmul_vv_f32m##lmul(r, r, vl);                       \
        auto y1 = __riscv_vfmv_v_f_f32m##lmul(0x1.5b2e76p-19f, vl);            \
        auto y2 = __riscv_vfmv_v_f_f32m##lmul(0x1.110df4p-7f, vl);             \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r2, c2, vl);                     \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, r2, c0, vl);                     \
        auto r4 = __riscv_vfmul_vv_f32m##lmul(r2, r2, vl);                     \
        auto r3 = __riscv_vfmul_vv_f32m##lmul(r2, r, vl);                      \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r4, y2, vl);                     \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r3, r, vl);                      \
        auto tmp = __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(y1);         \
        tmp = __riscv_vxor_vv_i32m##lmul(tmp, odd, vl);                        \
        return __riscv_vreinterpret_v_i32m##lmul##_f32m##lmul(tmp);            \
    }

REGISTER_RVV_KERNEL(COS_FLOAT32)
REGISTER_RVV_UNARY_OP(cos, float, cos_float32)

// cosh(v) = (exp(v) + exp(-v)) / 2
#if 0
#define COSH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t cosh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = exp_ps(vfneg_v_f32m##lmul(v, vl), vl);                        \
        auto sum = __riscv_vfadd_vv_f32m##lmul(a, b, vl);                      \
        return __riscv_vfdiv_vf_f32m##lmul(sum, 2.f, vl);                      \
    }
#else
#if 0
// max_ulp_error = 90164
#define COSH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t cosh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = __riscv_vfrec7_v_f32m##lmul(a, vl);                           \
        auto sum = __riscv_vfadd_vv_f32m##lmul(a, b, vl);                      \
        return __riscv_vfmul_vf_f32m##lmul(sum, 0.5f, vl);                     \
    }
#else
#define COSH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t cosh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = __riscv_vfrdiv_vf_f32m##lmul(a, 1.f, vl);                     \
        auto sum = __riscv_vfadd_vv_f32m##lmul(a, b, vl);                      \
        return __riscv_vfmul_vf_f32m##lmul(sum, 0.5f, vl);                     \
    }
#endif
#endif

REGISTER_RVV_KERNEL(COSH_FLOAT32)
REGISTER_RVV_UNARY_OP(cosh, float, cosh_float32)

// exp
#define EXP_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t exp_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return exp_ps(v, vl);                                                  \
    }

REGISTER_RVV_KERNEL(EXP_FLOAT32)
REGISTER_RVV_UNARY_OP(exp, float, exp_float32)

// floor
#define FLOOR_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t floor_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto vi = __riscv_vfcvt_x_f_v_i32m##lmul(v, vl);                       \
        auto vf = __riscv_vfcvt_f_x_v_f32m##lmul(vi, vl);                      \
        auto mask = __riscv_vmfgt_vv_f32m##lmul##_b##mlen(vf, v, vl);          \
        vf = __riscv_vfsub_vf_f32m##lmul##_m(mask, vf, 1.f, vl);               \
        return vf;                                                             \
    }
REGISTER_RVV_KERNEL(FLOOR_FLOAT32)
REGISTER_RVV_UNARY_OP(floor, float, floor_float32)

// log
#define LOG_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t log_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return log_ps(v, vl);                                                  \
    }

REGISTER_RVV_KERNEL(LOG_FLOAT32)
REGISTER_RVV_UNARY_OP(log, float, log_float32)

// neg
#define NEG_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t neg_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return __riscv_vfneg_v_f32m##lmul(v, vl);                              \
    }

REGISTER_RVV_KERNEL(NEG_FLOAT32)
REGISTER_RVV_UNARY_OP(neg, float, neg_float32)

// round
#define ROUND_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t round_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        return __riscv_vfcvt_f_x_v_f32m##lmul(                                 \
            __riscv_vfcvt_x_f_v_i32m##lmul(v, vl), vl);                        \
    }

REGISTER_RVV_KERNEL(ROUND_FLOAT32)
REGISTER_RVV_UNARY_OP(round, float, round_float32)

// rsqrt
#if 0
// max_ulp_error = 0
#define RSQRT_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t rsqrt_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        return __riscv_vfrdiv_vf_f32m##lmul(                                   \
            __riscv_vfsqrt_v_f32m##lmul(v, vl), 1.f, vl);                      \
    }
#else
#if 0
// max_ulp_error = 88880
#define RSQRT_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t rsqrt_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        return __riscv_vfrsqrt7_v_f32m##lmul(v, vl);                           \
    }
#else
#define RSQRT_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t rsqrt_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto one_point_five = __riscv_vfmv_v_f_f32m##lmul(1.5f, vl);           \
                                                                               \
        auto ux = __riscv_vreinterpret_v_f32m##lmul##_u32m##lmul(v);           \
        ux = __riscv_vsrl_vx_u32m##lmul(ux, 1, vl);                            \
        ux = __riscv_vrsub_vx_u32m##lmul(ux, 0x5f375a86, vl);                  \
        auto y = __riscv_vreinterpret_v_u32m##lmul##_f32m##lmul(ux);           \
                                                                               \
        auto y2 = __riscv_vfmul_vv_f32m##lmul(y, y, vl);                       \
        auto x = __riscv_vfmul_vf_f32m##lmul(v, -0.5f, vl);                    \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f32m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f32m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f32m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f32m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f32m##lmul(y, y2, vl);                            \
                                                                               \
        y2 = __riscv_vfmul_vv_f32m##lmul(y, y, vl);                            \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, x, one_point_five, vl);          \
        y = __riscv_vfmul_vv_f32m##lmul(y, y2, vl);                            \
        return y;                                                              \
    }
#endif
#endif

REGISTER_RVV_KERNEL(RSQRT_FLOAT32)
REGISTER_RVV_UNARY_OP(rsqrt, float, rsqrt_float32)

// sign
#define SIGN_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t sign_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto ret = __riscv_vfmv_v_f_f32m##lmul(0.f, vl);                       \
        auto gt_mask = __riscv_vmfgt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);      \
        ret = __riscv_vfmerge_vfm_f32m##lmul(ret, 1.f, gt_mask, vl);           \
        auto lt_mask = __riscv_vmflt_vf_f32m##lmul##_b##mlen(v, 0.f, vl);      \
        return __riscv_vfmerge_vfm_f32m##lmul(ret, -1.f, lt_mask, vl);         \
    }

REGISTER_RVV_KERNEL(SIGN_FLOAT32)
REGISTER_RVV_UNARY_OP(sign, float, sign_float32)

// sin
// from glibc 2.40: sysdeps/aarch64/fpu/sinf_advsimd.c
#define SIN_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t sin_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        auto c0 = __riscv_vfmv_v_f_f32m##lmul(-0x1.555548p-3f, vl);            \
        auto c2 = __riscv_vfmv_v_f_f32m##lmul(-0x1.9f42eap-13f, vl);           \
                                                                               \
        /* n = rint(|x|/pi) */                                                 \
        auto r = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        auto n = __riscv_vfmul_vf_f32m##lmul(r, 0x1.45f306p-2f, vl);           \
        auto sign = __riscv_vxor_vv_i32m##lmul(                                \
            __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(v),                 \
            __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(r), vl);            \
        auto ni = __riscv_vfcvt_x_f_v_i32m##lmul(n, vl);                       \
        n = __riscv_vfcvt_f_x_v_f32m##lmul(ni, vl);                            \
        auto odd = __riscv_vadd_vx_i32m##lmul(ni, 0x1.8p+23, vl);              \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, 0x1.921fb6p+1f, n, vl);           \
        odd = __riscv_vsll_vx_i32##m##lmul(odd, 31, vl);                       \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.777a5cp-24f, n, vl);         \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.ee59dap-49f, n, vl);         \
                                                                               \
        /* y = sin(r).  */                                                     \
        auto r2 = __riscv_vfmul_vv_f32m##lmul(r, r, vl);                       \
        auto y1 = __riscv_vfmv_v_f_f32m##lmul(0x1.5b2e76p-19f, vl);            \
        auto y2 = __riscv_vfmv_v_f_f32m##lmul(0x1.110df4p-7f, vl);             \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r2, c2, vl);                     \
        y2 = __riscv_vfmadd_vv_f32m##lmul(y2, r2, c0, vl);                     \
        auto r4 = __riscv_vfmul_vv_f32m##lmul(r2, r2, vl);                     \
        auto r3 = __riscv_vfmul_vv_f32m##lmul(r2, r, vl);                      \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r4, y2, vl);                     \
        sign = __riscv_vxor_vv_i32m##lmul(sign, odd, vl);                      \
        y1 = __riscv_vfmadd_vv_f32m##lmul(y1, r3, r, vl);                      \
        auto tmp = __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(y1);         \
        tmp = __riscv_vxor_vv_i32m##lmul(tmp, sign, vl);                       \
        return __riscv_vreinterpret_v_i32m##lmul##_f32m##lmul(tmp);            \
    }

REGISTER_RVV_KERNEL(SIN_FLOAT32)
REGISTER_RVV_UNARY_OP(sin, float, sin_float32)

// sinh(v) = (exp(v) - exp(-v)) / 2
#if 0
#define SINH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t sinh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = __riscv_vfrec7_v_f32m##lmul(a, vl);                           \
        return __riscv_vfmul_vf_f32m##lmul(                                    \
            __riscv_vfsub_vv_f32m##lmul(a, b, vl), 0.5f, vl);                  \
    }
#else
#define SINH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t sinh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = __riscv_vfrdiv_vf_f32m##lmul(a, 1.f, vl);                     \
        return __riscv_vfmul_vf_f32m##lmul(                                    \
            __riscv_vfsub_vv_f32m##lmul(a, b, vl), 0.5f, vl);                  \
    }
#endif

REGISTER_RVV_KERNEL(SINH_FLOAT32)
REGISTER_RVV_UNARY_OP(sinh, float, sinh_float32)

// sqrt
#define SQRT_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t sqrt_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        return __riscv_vfsqrt_v_f32m##lmul(v, vl);                             \
    }

REGISTER_RVV_KERNEL(SQRT_FLOAT32)
REGISTER_RVV_UNARY_OP(sqrt, float, sqrt_float32)

// square
#define SQUARE_FLOAT32(lmul, mlen)                                             \
    inline vfloat32m##lmul##_t square_float32(const vfloat32m##lmul##_t &v,    \
                                              const size_t vl) {               \
        return __riscv_vfmul_vv_f32m##lmul(v, v, vl);                          \
    }

REGISTER_RVV_KERNEL(SQUARE_FLOAT32)
REGISTER_RVV_UNARY_OP(square, float, square_float32)

// tanh
#define TANH_FLOAT32(lmul, mlen)                                               \
    inline vfloat32m##lmul##_t tanh_float32(const vfloat32m##lmul##_t &v,      \
                                            const size_t vl) {                 \
        return tanh_ps(v, vl);                                                 \
    }

REGISTER_RVV_KERNEL(TANH_FLOAT32)
REGISTER_RVV_UNARY_OP(tanh, float, tanh_float32)

// erf
#define ERF_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t erf_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        return erf_ps(v, vl);                                                  \
    }

REGISTER_RVV_KERNEL(ERF_FLOAT32)
REGISTER_RVV_UNARY_OP(erf, float, erf_float32)

// binary
#define RVV_BINARY_OP(op, dtype, vl, kernel)                                   \
    template <> struct op<ntt::vector<dtype, vl>, ntt::vector<dtype, vl>> {    \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v1,                           \
                   const ntt::vector<dtype, vl> &v2) const noexcept {          \
            return kernel(v1, v2, vl);                                         \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <> struct op<ntt::vector<dtype, vl>, dtype> {                     \
        ntt::vector<dtype, vl> operator()(const ntt::vector<dtype, vl> &v,     \
                                          const dtype &s) const noexcept {     \
            return kernel(v, s, vl);                                           \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <> struct op<dtype, ntt::vector<dtype, vl>> {                     \
        ntt::vector<dtype, vl>                                                 \
        operator()(const dtype &s,                                             \
                   const ntt::vector<dtype, vl> &v) const noexcept {           \
            return kernel(s, v, vl);                                           \
        };                                                                     \
    };

// binary op
#define REGISTER_RVV_BINARY_OP(op, dtype, kernel)                              \
    RVV_BINARY_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)          \
    RVV_BINARY_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)          \
    RVV_BINARY_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)          \
    RVV_BINARY_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

// add
#define ADD_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t add_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfadd_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t add_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfadd_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t add_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfadd_vf_f32m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_KERNEL(ADD_FLOAT32)
REGISTER_RVV_BINARY_OP(add, float, add_float32)

// sub
#define SUB_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t sub_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfsub_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t sub_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfsub_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t sub_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfrsub_vf_f32m##lmul(v, s, vl);                         \
    }

REGISTER_RVV_KERNEL(SUB_FLOAT32)
REGISTER_RVV_BINARY_OP(sub, float, sub_float32)

// mul
#define MUL_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t mul_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmul_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfmul_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfmul_vf_f32m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_KERNEL(MUL_FLOAT32)
REGISTER_RVV_BINARY_OP(mul, float, mul_float32)

// div
#define DIV_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t div_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfdiv_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t div_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfdiv_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t div_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfrdiv_vf_f32m##lmul(v, s, vl);                         \
    }

REGISTER_RVV_KERNEL(DIV_FLOAT32)
REGISTER_RVV_BINARY_OP(div, float, div_float32)

// mod
#define MOD_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t mod_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        auto quotient = __riscv_vfcvt_f_x_v_f32m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i32m##lmul(                                \
                __riscv_vfdiv_vv_f32m##lmul(v1, v2, vl), vl),                  \
            vl);                                                               \
        return __riscv_vfnmsub_vv_f32m##lmul(quotient, v2, v1, vl);            \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mod_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        auto quotient = __riscv_vfcvt_f_x_v_f32m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i32m##lmul(                                \
                __riscv_vfdiv_vf_f32m##lmul(v, s, vl), vl),                    \
            vl);                                                               \
        return __riscv_vfnmsub_vf_f32m##lmul(quotient, s, v, vl);              \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mod_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v2, const size_t vl) {      \
        auto v1 = __riscv_vfmv_v_f_f32m##lmul(s, vl);                          \
        auto quotient = __riscv_vfcvt_f_x_v_f32m##lmul(                        \
            __riscv_vfcvt_rtz_x_f_v_i32m##lmul(                                \
                __riscv_vfrdiv_vf_f32m##lmul(v2, s, vl), vl),                  \
            vl);                                                               \
        return __riscv_vfnmsub_vv_f32m##lmul(quotient, v2, v1, vl);            \
    }

REGISTER_RVV_KERNEL(MOD_FLOAT32)
REGISTER_RVV_BINARY_OP(mod, float, mod_float32)

// min
template <> struct min<float, float> {
    auto operator()(const float &s1, const float &s2) const noexcept {
        float ret;
        __asm("fmin.s %[ret], %[s1], %[s2];"
              : [ret] "=f"(ret)
              : [s1] "f"(s1), [s2] "f"(s2));
        return ret;
    }
};

#define MIN_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t min_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmin_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t min_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfmin_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t min_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfmin_vf_f32m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_KERNEL(MIN_FLOAT32)
REGISTER_RVV_BINARY_OP(min, float, min_float32)

// max
template <> struct max<float, float> {
    auto operator()(const float &s1, const float &s2) const noexcept {
        float ret;
        __asm("fmax.s %[ret], %[s1], %[s2];"
              : [ret] "=f"(ret)
              : [s1] "f"(s1), [s2] "f"(s2));
        return ret;
    }
};

#define MAX_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t max_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return __riscv_vfmax_vv_f32m##lmul(v1, v2, vl);                        \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t max_float32(const vfloat32m##lmul##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return __riscv_vfmax_vf_f32m##lmul(v, s, vl);                          \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t max_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v, const size_t vl) {       \
        return __riscv_vfmax_vf_f32m##lmul(v, s, vl);                          \
    }

REGISTER_RVV_KERNEL(MAX_FLOAT32)
REGISTER_RVV_BINARY_OP(max, float, max_float32)

// pow
#define POW_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t pow_float32(const vfloat32m##lmul##_t &v1,      \
                                           const vfloat32m##lmul##_t &v2,      \
                                           const size_t vl) {                  \
        return pow_ps(v1, v2, vl);                                             \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t pow_float32(const vfloat32m##lmul##_t &v1,      \
                                           const float &s, const size_t vl) {  \
        auto v2 = __riscv_vfmv_v_f_f32m##lmul(s, vl);                          \
        return pow_ps(v1, v2, vl);                                             \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t pow_float32(                                    \
        const float &s, const vfloat32m##lmul##_t &v2, const size_t vl) {      \
        auto v1 = __riscv_vfmv_v_f_f32m##lmul(s, vl);                          \
        return pow_ps(v1, v2, vl);                                             \
    }

REGISTER_RVV_KERNEL(POW_FLOAT32)
REGISTER_RVV_BINARY_OP(pow, float, pow_float32)

// floor_mod
#define FLOOR_MOD_INT32(lmul, mlen)                                            \
    inline vint32m##lmul##_t floor_mod_int32(const vint32m##lmul##_t &v1,      \
                                             const vint32m##lmul##_t &v2,      \
                                             const size_t vl) {                \
        auto remainder = __riscv_vrem_vv_i32m##lmul(v1, v2, vl);               \
        auto tmp = __riscv_vxor_vv_i32m##lmul(v1, v2, vl);                     \
        auto mask1 = __riscv_vmsne_vx_i32m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i32m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vv_i32m##lmul##_m(mask1, remainder, v2, vl);  \
        return remainder;                                                      \
    }                                                                          \
                                                                               \
    inline vint32m##lmul##_t floor_mod_int32(                                  \
        const vint32m##lmul##_t &v1, const int32_t &s, const size_t vl) {      \
        auto remainder = __riscv_vrem_vx_i32m##lmul(v1, s, vl);                \
        auto tmp = __riscv_vxor_vx_i32m##lmul(v1, s, vl);                      \
        auto mask1 = __riscv_vmsne_vx_i32m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i32m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vx_i32m##lmul##_m(mask1, remainder, s, vl);   \
        return remainder;                                                      \
    }                                                                          \
                                                                               \
    inline vint32m##lmul##_t floor_mod_int32(                                  \
        const int32_t &s, const vint32m##lmul##_t &v2, const size_t vl) {      \
        auto v1 = __riscv_vmv_v_x_i32m##lmul(s, vl);                           \
        auto remainder = __riscv_vrem_vv_i32m##lmul(v1, v2, vl);               \
        auto tmp = __riscv_vxor_vv_i32m##lmul(v1, v2, vl);                     \
        auto mask1 = __riscv_vmsne_vx_i32m##lmul##_b##mlen(remainder, 0, vl);  \
        auto mask2 = __riscv_vmslt_vx_i32m##lmul##_b##mlen(tmp, 0, vl);        \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        remainder = __riscv_vadd_vv_i32m##lmul##_m(mask1, remainder, v2, vl);  \
        return remainder;                                                      \
    }

REGISTER_RVV_KERNEL(FLOOR_MOD_INT32)
REGISTER_RVV_BINARY_OP(floor_mod, int32_t, floor_mod_int32)

// swish
// swish(v) = v / (exp(-v) + 1)
#define SWISH_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t swish_float32(const vfloat32m##lmul##_t &v,     \
                                             const size_t vl) {                \
        auto tmp = exp_ps(__riscv_vfneg_v_f32m##lmul(v, vl), vl);              \
        return __riscv_vfdiv_vv_f32m##lmul(                                    \
            v, __riscv_vfadd_vf_f32m##lmul(tmp, 1.f, vl), vl);                 \
    }

REGISTER_RVV_KERNEL(SWISH_FLOAT32)
REGISTER_RVV_UNARY_OP(swish, float, swish_float32)

// register swishb kernel
// swishb(v) = v / (exp(-v*beta) + 1)
#define SWISHB_FLOAT32(lmul, mlen)                                             \
    inline vfloat32m##lmul##_t swishb_float32(const vfloat32m##lmul##_t &v,    \
                                              float beta, const size_t vl) {   \
        auto tmp = __riscv_vfmul_vf_f32m##lmul(v, -beta, vl);                  \
        tmp = exp_ps(tmp, vl);                                                 \
        tmp = __riscv_vfadd_vf_f32m##lmul(tmp, 1.0f, vl);                      \
        return __riscv_vfdiv_vv_f32m##lmul(v, tmp, vl);                        \
    }

REGISTER_RVV_KERNEL(SWISHB_FLOAT32)

// register swishb op
#define RVV_SWISHB_OP(dtype, vl, kernel)                                       \
    template <> struct swishb<ntt::vector<dtype, vl>, dtype> {                 \
        ntt::vector<dtype, vl> operator()(const ntt::vector<dtype, vl> &v,     \
                                          const dtype &beta) const noexcept {  \
            return kernel(v, beta, vl);                                        \
        }                                                                      \
    };

#define REGISTER_RVV_SWISHB_OP(dtype, kernel)                                  \
    RVV_SWISHB_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)              \
    RVV_SWISHB_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)              \
    RVV_SWISHB_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)              \
    RVV_SWISHB_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_SWISHB_OP(float, swishb_float32)

// rigister outer_product op
#define RVV_OUTER_PRODUCT_OP(dtype, vl, lmul)                                  \
    template <>                                                                \
    struct outer_product<ntt::vector<dtype, vl>, ntt::vector<dtype, vl>> {     \
        auto operator()(const ntt::vector<dtype, vl> &v1,                      \
                        const ntt::vector<dtype, vl> &v2) const noexcept {     \
            vector<dtype, vl, vl> vout;                                        \
            if (vl == 4) {                                                     \
                vout(0) = __riscv_vfmul_vf_f32m##lmul(v2, v1(0), vl);          \
                vout(1) = __riscv_vfmul_vf_f32m##lmul(v2, v1(1), vl);          \
                vout(2) = __riscv_vfmul_vf_f32m##lmul(v2, v1(2), vl);          \
                vout(3) = __riscv_vfmul_vf_f32m##lmul(v2, v1(3), vl);          \
            } else {                                                           \
                for (size_t i = 0; i < vl; i++) {                              \
                    vout(i) = __riscv_vfmul_vf_f32m##lmul(v2, v1(i), vl);      \
                }                                                              \
            }                                                                  \
            return vout;                                                       \
        }                                                                      \
    };

#define REGISTER_RVV_OUTER_PRODUCT_OP(dtype)                                   \
    RVV_OUTER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), 1)            \
    RVV_OUTER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), 2)            \
    RVV_OUTER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), 4)            \
    RVV_OUTER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), 8)

REGISTER_RVV_OUTER_PRODUCT_OP(float)

// register inner_product kernel
#define INNER_PRODUCT_FLOAT32(lmul, mlen)                                      \
    inline float inner_product_float32(const vfloat32m##lmul##_t &v1,          \
                                       const vfloat32m##lmul##_t &v2,          \
                                       const size_t vl) {                      \
        auto zero = __riscv_vfmv_v_f_f32m1(0, vl);                             \
        auto tmp = __riscv_vfmul_vv_f32m##lmul(v1, v2, vl);                    \
        return __riscv_vfmv_f_s_f32m1_f32(                                     \
            __riscv_vfredusum_vs_f32m##lmul##_f32m1(tmp, zero, vl));           \
    }

REGISTER_RVV_KERNEL(INNER_PRODUCT_FLOAT32)

// register inner_product op
#define RVV_INNER_PRODUCT_OP(dtype, vl, kernel)                                \
    template <>                                                                \
    struct inner_product<ntt::vector<dtype, vl>, ntt::vector<dtype, vl>> {     \
        dtype operator()(const ntt::vector<dtype, vl> &v1,                     \
                         const ntt::vector<dtype, vl> &v2) const noexcept {    \
            return kernel(v1, v2, vl);                                         \
        }                                                                      \
    };

#define REGISTER_RVV_INNER_PRODUCT_OP(dtype, kernel)                           \
    RVV_INNER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)       \
    RVV_INNER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)       \
    RVV_INNER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)       \
    RVV_INNER_PRODUCT_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_INNER_PRODUCT_OP(float, inner_product_float32)

// register mul_add kernel
#if 0
#define MUL_ADD_FLOAT32(lmul, mlen)                                            \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const vfloat32m##lmul##_t &v1, const vfloat32m##lmul##_t &v2,          \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmadd_vv_f32m##lmul(v1, v2, v3, vl);                   \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const vfloat32m##lmul##_t &v1, const float &s2,                        \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmadd_vf_f32m##lmul(v1, s2, v3, vl);                   \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const float &s1, const vfloat32m##lmul##_t &v2,                        \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmadd_vf_f32m##lmul(v2, s1, v3, vl);                   \
    }
#else
#define MUL_ADD_FLOAT32(lmul, mlen)                                            \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const vfloat32m##lmul##_t &v1, const vfloat32m##lmul##_t &v2,          \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmacc_vv_f32m##lmul(v3, v1, v2, vl);                   \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const vfloat32m##lmul##_t &v1, const float &s2,                        \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmacc_vf_f32m##lmul(v3, s2, v1, vl);                   \
    }                                                                          \
                                                                               \
    inline vfloat32m##lmul##_t mul_add_float32(                                \
        const float &s1, const vfloat32m##lmul##_t &v2,                        \
        const vfloat32m##lmul##_t &v3, const size_t vl) {                      \
        return __riscv_vfmacc_vf_f32m##lmul(v3, s1, v2, vl);                   \
    }
#endif

REGISTER_RVV_KERNEL(MUL_ADD_FLOAT32)

// register mul_add op
#define RVV_MUL_ADD_OP(dtype, vl, kernel)                                      \
    template <>                                                                \
    struct mul_add<ntt::vector<dtype, vl>, ntt::vector<dtype, vl>,             \
                   ntt::vector<dtype, vl>> {                                   \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v1,                           \
                   const ntt::vector<dtype, vl> &v2,                           \
                   const ntt::vector<dtype, vl> &v3) const noexcept {          \
            return kernel(v1, v2, v3, vl);                                     \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <>                                                                \
    struct mul_add<ntt::vector<dtype, vl>, dtype, ntt::vector<dtype, vl>> {    \
        ntt::vector<dtype, vl>                                                 \
        operator()(const ntt::vector<dtype, vl> &v1, const dtype &s2,          \
                   const ntt::vector<dtype, vl> &v3) const noexcept {          \
            return kernel(v1, s2, v3, vl);                                     \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <>                                                                \
    struct mul_add<dtype, ntt::vector<dtype, vl>, ntt::vector<dtype, vl>> {    \
        ntt::vector<dtype, vl>                                                 \
        operator()(const dtype &s1, const ntt::vector<dtype, vl> &v2,          \
                   const ntt::vector<dtype, vl> &v3) const noexcept {          \
            return kernel(s1, v2, v3, vl);                                     \
        }                                                                      \
    };

#define REGISTER_RVV_MUL_ADD_OP(dtype, kernel)                                 \
    RVV_MUL_ADD_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)             \
    RVV_MUL_ADD_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)             \
    RVV_MUL_ADD_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)             \
    RVV_MUL_ADD_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_MUL_ADD_OP(float, mul_add_float32)

template <bool AccC>
struct mma<AccC, false, ntt::vector<float, 1, 4>, ntt::vector<float, 4, 4>,
           ntt::vector<float, 1, 4>> {
    ntt::vector<float, 1, 4>
    operator()(const ntt::vector<float, 1, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 1, 4> &v3) const noexcept {
        auto output = v3;
        auto t0 = AccC ? ntt::mul_add(lhs(0, 0), rhs(0), output(0))
                       : ntt::mul(lhs(0, 0), rhs(0));
        auto t1 = ntt::mul(lhs(0, 1), rhs(1));
        t0 = ntt::mul_add(lhs(0, 2), rhs(2), t0);
        t1 = ntt::mul_add(lhs(0, 3), rhs(3), t1);
        output(0) = ntt::add(t0, t1);
        return output;
    }
};

template <bool AccC>
struct mma<AccC, false, ntt::vector<float, 1, 32>, ntt::vector<float, 32, 32>,
           ntt::vector<float, 1, 32>> {
    ntt::vector<float, 1, 32>
    operator()(const ntt::vector<float, 1, 32> &lhs,
               const ntt::vector<float, 32, 32> &rhs,
               const ntt::vector<float, 1, 32> &v3) const noexcept {
        auto output = v3;

        auto t0 = AccC ? ntt::mul_add(lhs(0, 0), rhs(0), output(0))
                       : ntt::mul(lhs(0, 0), rhs(0));
        auto t1 = ntt::mul(lhs(0, 1), rhs(1));
        t0 = ntt::mul_add(lhs(0, 2), rhs(2), t0);
        t1 = ntt::mul_add(lhs(0, 3), rhs(3), t1);

        t0 = ntt::mul_add(lhs(0, 4), rhs(4), t0);
        t1 = ntt::mul_add(lhs(0, 5), rhs(5), t1);
        t0 = ntt::mul_add(lhs(0, 6), rhs(6), t0);
        t1 = ntt::mul_add(lhs(0, 7), rhs(7), t1);

        t0 = ntt::mul_add(lhs(0, 8), rhs(8), t0);
        t1 = ntt::mul_add(lhs(0, 9), rhs(9), t1);
        t0 = ntt::mul_add(lhs(0, 10), rhs(10), t0);
        t1 = ntt::mul_add(lhs(0, 11), rhs(11), t1);

        t0 = ntt::mul_add(lhs(0, 12), rhs(12), t0);
        t1 = ntt::mul_add(lhs(0, 13), rhs(13), t1);
        t0 = ntt::mul_add(lhs(0, 14), rhs(14), t0);
        t1 = ntt::mul_add(lhs(0, 15), rhs(15), t1);

        t0 = ntt::mul_add(lhs(0, 16), rhs(16), t0);
        t1 = ntt::mul_add(lhs(0, 17), rhs(17), t1);
        t0 = ntt::mul_add(lhs(0, 18), rhs(18), t0);
        t1 = ntt::mul_add(lhs(0, 19), rhs(19), t1);

        t0 = ntt::mul_add(lhs(0, 20), rhs(20), t0);
        t1 = ntt::mul_add(lhs(0, 21), rhs(21), t1);
        t0 = ntt::mul_add(lhs(0, 22), rhs(22), t0);
        t1 = ntt::mul_add(lhs(0, 23), rhs(23), t1);

        t0 = ntt::mul_add(lhs(0, 24), rhs(24), t0);
        t1 = ntt::mul_add(lhs(0, 25), rhs(25), t1);
        t0 = ntt::mul_add(lhs(0, 26), rhs(26), t0);
        t1 = ntt::mul_add(lhs(0, 27), rhs(27), t1);

        t0 = ntt::mul_add(lhs(0, 28), rhs(28), t0);
        t1 = ntt::mul_add(lhs(0, 29), rhs(29), t1);
        t0 = ntt::mul_add(lhs(0, 30), rhs(30), t0);
        t1 = ntt::mul_add(lhs(0, 31), rhs(31), t1);

        output(0) = ntt::add(t0, t1);
        return output;
    }
};

template <bool AccC>
struct mma<AccC, false, ntt::vector<float, 4, 4>, ntt::vector<float, 4, 4>,
           ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4>
    operator()(const ntt::vector<float, 4, 4> &lhs,
               const ntt::vector<float, 4, 4> &rhs,
               const ntt::vector<float, 4, 4> &v3) const noexcept {
        auto output = v3;
        ntt::fixed_tensor_alike_t<ntt::vector<float, 4>, 1, 4> lhs_2d[4]{
            {{lhs(0)}},
            {{lhs(1)}},
            {{lhs(2)}},
            {{lhs(3)}},
        };
        ntt::fixed_tensor_alike_t<ntt::vector<float, 4>, 1, 4> output_2d[4]{
            {{v3(0)}},
            {{v3(1)}},
            {{v3(2)}},
            {{v3(3)}},
        };

        output_2d[0] = ntt::mma<AccC>(lhs_2d[0], rhs, output_2d[0]);
        output_2d[1] = ntt::mma<AccC>(lhs_2d[1], rhs, output_2d[1]);
        output_2d[2] = ntt::mma<AccC>(lhs_2d[2], rhs, output_2d[2]);
        output_2d[3] = ntt::mma<AccC>(lhs_2d[3], rhs, output_2d[3]);

        output(0) = output_2d[0](0);
        output(1) = output_2d[1](0);
        output(2) = output_2d[2](0);
        output(3) = output_2d[3](0);

        return output;
    }
};

template <bool AccC>
struct mma<AccC, false, ntt::vector<float, 32, 32>, ntt::vector<float, 32, 32>,
           ntt::vector<float, 32, 32>> {
    ntt::vector<float, 32, 32>
    operator()(const ntt::vector<float, 32, 32> &lhs,
               const ntt::vector<float, 32, 32> &rhs,
               const ntt::vector<float, 32, 32> &v3) const noexcept {
        auto output = v3;
        ntt::fixed_tensor_alike_t<ntt::vector<float, 32>, 1, 32> lhs_2d[]{
            {{lhs(0)}},  {{lhs(1)}},  {{lhs(2)}},  {{lhs(3)}},  {{lhs(4)}},
            {{lhs(5)}},  {{lhs(6)}},  {{lhs(7)}},  {{lhs(8)}},  {{lhs(9)}},
            {{lhs(10)}}, {{lhs(11)}}, {{lhs(12)}}, {{lhs(13)}}, {{lhs(14)}},
            {{lhs(15)}}, {{lhs(16)}}, {{lhs(17)}}, {{lhs(18)}}, {{lhs(19)}},
            {{lhs(20)}}, {{lhs(21)}}, {{lhs(22)}}, {{lhs(23)}}, {{lhs(24)}},
            {{lhs(25)}}, {{lhs(26)}}, {{lhs(27)}}, {{lhs(28)}}, {{lhs(29)}},
            {{lhs(30)}}, {{lhs(31)}}};

        ntt::fixed_tensor_alike_t<ntt::vector<float, 32>, 1, 32> output_2d[]{
            {{v3(0)}},  {{v3(1)}},  {{v3(2)}},  {{v3(3)}},  {{v3(4)}},
            {{v3(5)}},  {{v3(6)}},  {{v3(7)}},  {{v3(8)}},  {{v3(9)}},
            {{v3(10)}}, {{v3(11)}}, {{v3(12)}}, {{v3(13)}}, {{v3(14)}},
            {{v3(15)}}, {{v3(16)}}, {{v3(17)}}, {{v3(18)}}, {{v3(19)}},
            {{v3(20)}}, {{v3(21)}}, {{v3(22)}}, {{v3(23)}}, {{v3(24)}},
            {{v3(25)}}, {{v3(26)}}, {{v3(27)}}, {{v3(28)}}, {{v3(29)}},
            {{v3(30)}}, {{v3(31)}}};

        output_2d[0] = ntt::mma<AccC>(lhs_2d[0], rhs, output_2d[0]);
        output_2d[1] = ntt::mma<AccC>(lhs_2d[1], rhs, output_2d[1]);
        output_2d[2] = ntt::mma<AccC>(lhs_2d[2], rhs, output_2d[2]);
        output_2d[3] = ntt::mma<AccC>(lhs_2d[3], rhs, output_2d[3]);
        output_2d[4] = ntt::mma<AccC>(lhs_2d[4], rhs, output_2d[4]);
        output_2d[5] = ntt::mma<AccC>(lhs_2d[5], rhs, output_2d[5]);
        output_2d[6] = ntt::mma<AccC>(lhs_2d[6], rhs, output_2d[6]);
        output_2d[7] = ntt::mma<AccC>(lhs_2d[7], rhs, output_2d[7]);

        output_2d[8] = ntt::mma<AccC>(lhs_2d[8], rhs, output_2d[8]);
        output_2d[9] = ntt::mma<AccC>(lhs_2d[9], rhs, output_2d[9]);
        output_2d[10] = ntt::mma<AccC>(lhs_2d[10], rhs, output_2d[10]);
        output_2d[11] = ntt::mma<AccC>(lhs_2d[11], rhs, output_2d[11]);
        output_2d[12] = ntt::mma<AccC>(lhs_2d[12], rhs, output_2d[12]);
        output_2d[13] = ntt::mma<AccC>(lhs_2d[13], rhs, output_2d[13]);
        output_2d[14] = ntt::mma<AccC>(lhs_2d[14], rhs, output_2d[14]);
        output_2d[15] = ntt::mma<AccC>(lhs_2d[15], rhs, output_2d[15]);

        output_2d[16] = ntt::mma<AccC>(lhs_2d[16], rhs, output_2d[16]);
        output_2d[17] = ntt::mma<AccC>(lhs_2d[17], rhs, output_2d[17]);
        output_2d[18] = ntt::mma<AccC>(lhs_2d[18], rhs, output_2d[18]);
        output_2d[19] = ntt::mma<AccC>(lhs_2d[19], rhs, output_2d[19]);
        output_2d[20] = ntt::mma<AccC>(lhs_2d[20], rhs, output_2d[20]);
        output_2d[21] = ntt::mma<AccC>(lhs_2d[21], rhs, output_2d[21]);
        output_2d[22] = ntt::mma<AccC>(lhs_2d[22], rhs, output_2d[22]);
        output_2d[23] = ntt::mma<AccC>(lhs_2d[23], rhs, output_2d[23]);

        output_2d[24] = ntt::mma<AccC>(lhs_2d[24], rhs, output_2d[24]);
        output_2d[25] = ntt::mma<AccC>(lhs_2d[25], rhs, output_2d[25]);
        output_2d[26] = ntt::mma<AccC>(lhs_2d[26], rhs, output_2d[26]);
        output_2d[27] = ntt::mma<AccC>(lhs_2d[27], rhs, output_2d[27]);
        output_2d[28] = ntt::mma<AccC>(lhs_2d[28], rhs, output_2d[28]);
        output_2d[29] = ntt::mma<AccC>(lhs_2d[29], rhs, output_2d[29]);
        output_2d[30] = ntt::mma<AccC>(lhs_2d[30], rhs, output_2d[30]);
        output_2d[31] = ntt::mma<AccC>(lhs_2d[31], rhs, output_2d[31]);

        output(0) = output_2d[0](0);
        output(1) = output_2d[1](0);
        output(2) = output_2d[2](0);
        output(3) = output_2d[3](0);
        output(4) = output_2d[4](0);
        output(5) = output_2d[5](0);
        output(6) = output_2d[6](0);
        output(7) = output_2d[7](0);

        output(8) = output_2d[8](0);
        output(9) = output_2d[9](0);
        output(10) = output_2d[10](0);
        output(11) = output_2d[11](0);
        output(12) = output_2d[12](0);
        output(13) = output_2d[13](0);
        output(14) = output_2d[14](0);
        output(15) = output_2d[15](0);

        output(16) = output_2d[16](0);
        output(17) = output_2d[17](0);
        output(18) = output_2d[18](0);
        output(19) = output_2d[19](0);
        output(20) = output_2d[20](0);
        output(21) = output_2d[21](0);
        output(22) = output_2d[22](0);
        output(23) = output_2d[23](0);

        output(24) = output_2d[24](0);
        output(25) = output_2d[25](0);
        output(26) = output_2d[26](0);
        output(27) = output_2d[27](0);
        output(28) = output_2d[28](0);
        output(29) = output_2d[29](0);
        output(30) = output_2d[30](0);
        output(31) = output_2d[31](0);

        return output;
    }
};

// register reduce_sum kernel
#define REDUCE_ADD_FLOAT32(lmul, mlen)                                         \
    inline float reduce_add_float32(const vfloat32m##lmul##_t &v,              \
                                    const size_t vl) {                         \
        auto scalar = __riscv_vfmv_v_f_f32m1(0.f, vl);                         \
        auto dest = __riscv_vfredusum_vs_f32m##lmul##_f32m1(v, scalar, vl);    \
        return __riscv_vfmv_f_s_f32m1_f32(dest);                               \
    }

// register reduce_max kernel
#define REDUCE_MAX_FLOAT32(lmul, mlen)                                         \
    inline float reduce_max_float32(const vfloat32m##lmul##_t &v,              \
                                    const size_t vl) {                         \
        float lowest = std::numeric_limits<float>::lowest();                   \
        auto scalar = __riscv_vfmv_v_f_f32m1(lowest, vl);                      \
        auto dest = __riscv_vfredmax_vs_f32m##lmul##_f32m1(v, scalar, vl);     \
        return __riscv_vfmv_f_s_f32m1_f32(dest);                               \
    }

// register reduce_min kernel
#define REDUCE_MIN_FLOAT32(lmul, mlen)                                         \
    inline float reduce_min_float32(const vfloat32m##lmul##_t &v,              \
                                    const size_t vl) {                         \
        float max = std::numeric_limits<float>::max();                         \
        auto scalar = __riscv_vfmv_v_f_f32m1(max, vl);                         \
        auto dest = __riscv_vfredmin_vs_f32m##lmul##_f32m1(v, scalar, vl);     \
        return __riscv_vfmv_f_s_f32m1_f32(dest);                               \
    }

REGISTER_RVV_KERNEL(REDUCE_ADD_FLOAT32)
REGISTER_RVV_KERNEL(REDUCE_MAX_FLOAT32)
REGISTER_RVV_KERNEL(REDUCE_MIN_FLOAT32)

// register reduce op
#define RVV_REDUCE_OP(op, dtype, vl, kernel)                                   \
    template <> struct reduce<op, dtype, ntt::vector<dtype, vl>> {             \
        dtype operator()(const ntt::vector<dtype, vl> &v) const noexcept {     \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

#define REGISTER_RVV_REDUCE_OP(op, dtype, kernel)                              \
    RVV_REDUCE_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)          \
    RVV_REDUCE_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)          \
    RVV_REDUCE_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)          \
    RVV_REDUCE_OP(op, dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_REDUCE_OP(add, float, reduce_add_float32)
REGISTER_RVV_REDUCE_OP(max, float, reduce_max_float32)
REGISTER_RVV_REDUCE_OP(min, float, reduce_min_float32)

// register clamp kernel
#define CLAMP_FLOAT32(lmul, mlen)                                              \
    inline vfloat32m##lmul##_t clamp_float32(                                  \
        const vfloat32m##lmul##_t &v, const float &min, const float &max,      \
        const size_t vl) {                                                     \
        auto ret = __riscv_vfmax_vf_f32m##lmul(v, min, vl);                    \
        return __riscv_vfmin_vf_f32m##lmul(ret, max, vl);                      \
    }

REGISTER_RVV_KERNEL(CLAMP_FLOAT32)

// register clamp op
#define RVV_CLAMP_OP(dtype, vl, kernel)                                        \
    template <> struct clamp<ntt::vector<dtype, vl>, dtype> {                  \
        ntt::vector<dtype, vl> operator()(const ntt::vector<dtype, vl> &v,     \
                                          const dtype &min,                    \
                                          const dtype &max) const noexcept {   \
            return kernel(v, min, max, vl);                                    \
        }                                                                      \
    };

#define REGISTER_RVV_CLAMP_OP(dtype, kernel)                                   \
    RVV_CLAMP_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 1), kernel)               \
    RVV_CLAMP_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 2), kernel)               \
    RVV_CLAMP_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 4), kernel)               \
    RVV_CLAMP_OP(dtype, NTT_VL(sizeof(dtype) * 8, *, 8), kernel)

REGISTER_RVV_CLAMP_OP(float, clamp_float32)

// register cast kernel
#define CAST_FLOAT32_INT32(lmul, mlen)                                         \
    inline vint32m##lmul##_t cast_float32_int32(const vfloat32m##lmul##_t &v,  \
                                                const size_t vl) {             \
        return __riscv_vfcvt_rtz_x_f_v_i32m##lmul(v, vl);                      \
    }

#define CAST_INT32_FLOAT32(lmul, mlen)                                         \
    inline vfloat32m##lmul##_t cast_int32_float32(const vint32m##lmul##_t &v,  \
                                                  const size_t vl) {           \
        return __riscv_vfcvt_f_x_v_f32m##lmul(v, vl);                          \
    }

#define CAST_FLOAT32_UINT32(lmul, mlen)                                        \
    inline vuint32m##lmul##_t cast_float32_uint32(                             \
        const vfloat32m##lmul##_t &v, const size_t vl) {                       \
        return __riscv_vfcvt_xu_f_v_u32m##lmul(v, vl);                         \
    }

#define CAST_UINT32_FLOAT32(lmul, mlen)                                        \
    inline vfloat32m##lmul##_t cast_uint32_float32(                            \
        const vuint32m##lmul##_t &v, const size_t vl) {                        \
        return __riscv_vfcvt_f_xu_v_f32m##lmul(v, vl);                         \
    }

#define CAST_FLOAT32_BOOL(lmul32, lmul16, lmul8, mlen)                         \
    inline vuint8m##lmul8##_t cast_float32_bool(                               \
        const vfloat32m##lmul32##_t &v, const size_t vl) {                     \
        auto zero = __riscv_vmv_v_x_u8m##lmul8(0, vl);                         \
        auto mask = __riscv_vmfne_vf_f32m##lmul32##_b##mlen(v, 0.f, vl);       \
        return __riscv_vmerge_vxm_u8m##lmul8(zero, 1, mask, vl);               \
    }

#define CAST_BOOL_FLOAT32(lmul1, lmul2, mlen)                                  \
    inline vfloat32m##lmul2##_t cast_bool_float32(const vuint8m##lmul1##_t &v, \
                                                  const size_t vl) {           \
        auto zero = __riscv_vfmv_v_f_f32m##lmul2(0.f, vl);                     \
        auto mask = __riscv_vmsne_vx_u8m##lmul1##_b##mlen(v, 0, vl);           \
        return __riscv_vfmerge_vfm_f32m##lmul2(zero, 1.f, mask, vl);           \
    }

#define CAST_FLOAT32_FLOAT8_E4M3(lmul32, lmul16, lmul8, mlen)                  \
    inline vuint8m##lmul8##_t cast_float32_float8_e4m3(                        \
        const vfloat32m##lmul32##_t &v, const size_t vl) {                     \
        auto u8_128 = __riscv_vmv_v_x_u8m##lmul8(0x80, vl);                    \
        auto u32_zero = __riscv_vmv_v_x_u32m##lmul32(0, vl);                   \
        auto u32_one = __riscv_vmv_v_x_u32m##lmul32(1, vl);                    \
        auto s = __riscv_vreinterpret_v_f32m##lmul32##_u32m##lmul32(v);        \
        s = __riscv_vand_vx_u32m##lmul32(s, 0x7fffffff, vl);                   \
        auto mask1 = __riscv_vmfge_vf_f32m##lmul32##_b##mlen(v, 0.f, vl);      \
        auto sign = __riscv_vmerge_vxm_u8m##lmul8(u8_128, 0x00, mask1, vl);    \
        auto u32_tmp = __riscv_vsrl_vx_u32m##lmul32(s, 23, vl);                \
        auto i32_exp =                                                         \
            __riscv_vreinterpret_v_u32m##lmul32##_i32m##lmul32(u32_tmp);       \
        i32_exp = __riscv_vsub_vx_i32m##lmul32(i32_exp, 127, vl);              \
        auto i8_exp = __riscv_vncvt_x_x_w_i8m##lmul8(                          \
            __riscv_vncvt_x_x_w_i16m##lmul16(i32_exp, vl), vl);                \
        auto mantissa = __riscv_vand_vx_u32m##lmul32(s, 0x7fffff, vl);         \
        auto i8_tmp = __riscv_vadd_vx_i8m##lmul8(i8_exp, 7, vl);               \
        i8_tmp = __riscv_vsll_vx_i8m##lmul8(i8_tmp, 3, vl);                    \
        mask1 = __riscv_vmslt_vx_i8m##lmul8##_b##mlen(i8_exp, -6, vl);         \
        auto rshift = __riscv_vreinterpret_v_i32m##lmul32##_u32m##lmul32(      \
            __riscv_vrsub_vx_i32m##lmul32(i32_exp, -6, vl));                   \
        auto mask2 = __riscv_vmsltu_vx_u32m##lmul32##_b##mlen(rshift, 32, vl); \
        mask2 = __riscv_vmand_mm_b##mlen(mask2, mask1, vl);                    \
        auto u32_tmp2 = __riscv_vor_vx_u32m##lmul32(mantissa, 1 << 23, vl);    \
        u32_tmp = __riscv_vsll_vv_u32m##lmul32(u32_one, rshift, vl);           \
        u32_tmp = __riscv_vsub_vx_u32m##lmul32(u32_tmp, 1, vl);                \
        u32_tmp = __riscv_vand_vv_u32m##lmul32(u32_tmp, u32_tmp2, vl);         \
        auto sticky_bit =                                                      \
            __riscv_vmerge_vvm_u32m##lmul32(u32_zero, u32_tmp, mask2, vl);     \
        u32_tmp2 = __riscv_vsrl_vv_u32m##lmul32(u32_tmp2, rshift, vl);         \
        mantissa =                                                             \
            __riscv_vmerge_vvm_u32m##lmul32(mantissa, u32_tmp2, mask1, vl);    \
        u32_tmp = __riscv_vsrl_vx_u32m##lmul32(mantissa, 23 - 3, vl);          \
        u32_tmp2 = __riscv_vand_vx_u32m##lmul32(u32_tmp, (1 << 3) - 1, vl);    \
        u32_tmp =                                                              \
            __riscv_vmerge_vvm_u32m##lmul32(u32_tmp, u32_tmp2, mask1, vl);     \
        auto u8_tmp = __riscv_vncvt_x_x_w_u8m##lmul8(                          \
            __riscv_vncvt_x_x_w_u16m##lmul16(u32_tmp, vl), vl);                \
        auto u = __riscv_vor_vv_u8m##lmul8(                                    \
            __riscv_vreinterpret_v_i8m##lmul8##_u8m##lmul8(i8_tmp), u8_tmp,    \
            vl);                                                               \
        u = __riscv_vmerge_vvm_u8m##lmul8(u, u8_tmp, mask1, vl);               \
        u32_tmp = __riscv_vsrl_vx_u32m##lmul32(mantissa, 23 - 3 - 1, vl);      \
        auto round_bit = __riscv_vand_vx_u32m##lmul32(u32_tmp, 1, vl);         \
        u32_tmp = __riscv_vand_vx_u32m##lmul32(mantissa,                       \
                                               (1 << (23 - 3 - 1)) - 1, vl);   \
        sticky_bit = __riscv_vor_vv_u32m##lmul32(sticky_bit, u32_tmp, vl);     \
        u8_tmp = __riscv_vand_vx_u8m##lmul8(u, 1, vl);                         \
        mask1 = __riscv_vmsne_vx_u8m##lmul8##_b##mlen(u8_tmp, 0, vl);          \
        mask2 = __riscv_vmsne_vx_u32m##lmul32##_b##mlen(sticky_bit, 0, vl);    \
        mask1 = __riscv_vmor_mm_b##mlen(mask1, mask2, vl);                     \
        mask2 = __riscv_vmsne_vx_u32m##lmul32##_b##mlen(round_bit, 0, vl);     \
        mask1 = __riscv_vmand_mm_b##mlen(mask1, mask2, vl);                    \
        u8_tmp = __riscv_vadd_vx_u8m##lmul8(u, 1, vl);                         \
        u = __riscv_vmerge_vvm_u8m##lmul8(u, u8_tmp, mask1, vl);               \
        u = __riscv_vminu_vx_u8m##lmul8(u, 0x7e, vl);                          \
        mask1 = __riscv_vmsgtu_vx_u32m##lmul32##_b##mlen(s, 0x7f800000, vl);   \
        u = __riscv_vmerge_vxm_u8m##lmul8(u, 0x7f, mask1, vl);                 \
        mask1 = __riscv_vmseq_vx_u32m##lmul32##_b##mlen(s, 0x7f800000, vl);    \
        mask2 = __riscv_vmsgt_vx_i8m##lmul8##_b##mlen(i8_exp, 8, vl);          \
        mask1 = __riscv_vmor_mm_b##mlen(mask1, mask2, vl);                     \
        u = __riscv_vmerge_vxm_u8m##lmul8(u, 0x7e, mask1, vl);                 \
        return __riscv_vor_vv_u8m##lmul8(u, sign, vl);                         \
    }

REGISTER_RVV_KERNEL(CAST_FLOAT32_INT32)
REGISTER_RVV_KERNEL(CAST_INT32_FLOAT32)
REGISTER_RVV_KERNEL(CAST_FLOAT32_UINT32)
REGISTER_RVV_KERNEL(CAST_UINT32_FLOAT32)
REGISTER_RVV_KERNEL_4_1(CAST_FLOAT32_BOOL)
REGISTER_RVV_KERNEL_1_4(CAST_BOOL_FLOAT32)
REGISTER_RVV_KERNEL_4_1(CAST_FLOAT32_FLOAT8_E4M3)

// register cast op
#define RVV_CAST_OP(from_dtype, to_dtype, vl, kernel)                          \
    template <>                                                                \
    struct cast<ntt::vector<from_dtype, vl>, ntt::vector<to_dtype, vl>> {      \
        ntt::vector<to_dtype, vl>                                              \
        operator()(const ntt::vector<from_dtype, vl> &v) const noexcept {      \
            return kernel(v, vl);                                              \
        }                                                                      \
    };

#define REGISTER_RVV_CAST_OP(from, to, kernel)                                 \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 1), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 2), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 4), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 8), kernel)

#define REGISTER_RVV_CAST_OP_1_4(from, to, kernel)                             \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, /, 8), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, /, 4), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, /, 2), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 1), kernel)              \
    RVV_CAST_OP(from, to, NTT_VL(sizeof(from) * 8, *, 2), kernel)

REGISTER_RVV_CAST_OP(float, int, cast_float32_int32)
REGISTER_RVV_CAST_OP(int, float, cast_int32_float32)
REGISTER_RVV_CAST_OP(float, unsigned int, cast_float32_uint32)
REGISTER_RVV_CAST_OP(unsigned int, float, cast_uint32_float32)
REGISTER_RVV_CAST_OP(float, bool, cast_float32_bool)
REGISTER_RVV_CAST_OP_1_4(bool, float, cast_bool_float32)
REGISTER_RVV_CAST_OP(float, float_e4m3_t, cast_float32_float8_e4m3)

#endif
} // namespace nncase::ntt::ops