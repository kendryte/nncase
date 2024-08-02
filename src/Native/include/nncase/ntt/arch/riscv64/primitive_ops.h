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
#include "../../primitive_ops.h"
#include "rvv_mathfun.h"

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace nncase::ntt::ops {

#ifdef __riscv_vector
#define IMPL_RVV_WITH_LMULS(func) func(1, 32) func(2, 16) func(4, 8) func(8, 4)
#define REGISTER_RVV_WITH_VLENS(func, op) func(op, 128) func(op, 4096)

#define RVV_UNARY_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel)         \
    template <> struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {       \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v)       \
            const noexcept {                                                   \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto pin = reinterpret_cast<const dtype *>(v.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input = vle##sew##_v_##dtype_prefix##sew##m##lmul(            \
                pin, NTT_VL(vlen, sew, lmul));                                 \
            auto output = kernel(input, NTT_VL(vlen, sew, lmul));              \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
            return vout;                                                       \
        }                                                                      \
    };

// unary with float
#define REGISTER_RVV_UNARY_OP_FLOAT32(OP, vlen)                                \
    RVV_UNARY_OP(OP, float, f, vlen, 32, 1, OP##_float32)                      \
    RVV_UNARY_OP(OP, float, f, vlen, 32, 2, OP##_float32)                      \
    RVV_UNARY_OP(OP, float, f, vlen, 32, 4, OP##_float32)                      \
    RVV_UNARY_OP(OP, float, f, vlen, 32, 8, OP##_float32)

// abs
#define ABS_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t abs_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return vfabs_v_f32m##LMUL(v, vl);                                      \
    }

IMPL_RVV_WITH_LMULS(ABS_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, abs)

// acos
#if 0
// max_ulp_error = 789 on c908
// porting from https://developer.download.nvidia.cn/cg/acos.html
#define ACOS_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t acos_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto c2 = vfmv_v_f_f32m##LMUL(0.0742610f, vl);                         \
        auto c3 = vfmv_v_f_f32m##LMUL(-0.2121144f, vl);                        \
        auto c4 = vfmv_v_f_f32m##LMUL(1.5707288f, vl);                         \
        auto c5 = vfmv_v_f_f32m##LMUL(3.14159265358979f, vl);                  \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);                 \
        auto sroot =                                                           \
            vfsqrt_v_f32m##LMUL(vfrsub_vf_f32m##LMUL(x, 1.f, vl), vl);         \
        auto ret = vmv_v_v_f32m##LMUL(x, vl);                                  \
        ret = vfmadd_vf_f32m##LMUL(ret, -0.0187293f, c2, vl);                  \
        ret = vfmadd_vv_f32m##LMUL(ret, x, c3, vl);                            \
        ret = vfmadd_vv_f32m##LMUL(ret, x, c4, vl);                            \
        ret = vfmul_vv_f32m##LMUL(ret, sroot, vl);                             \
        return vfmadd_vf_f32m##LMUL##_m(mask, ret, -1.f, c5, vl);              \
    }
#else
// from glibc 2.40: sysdeps/aarch64/fpu/acosf_advsimd.c
#define ACOS_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t acos_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto zero = vfmv_v_f_f32m##LMUL(0.f, vl);                              \
        auto half = vfmv_v_f_f32m##LMUL(0.5f, vl);                             \
        auto one = vfmv_v_f_f32m##LMUL(1.f, vl);                               \
        auto two = vfmv_v_f_f32m##LMUL(2.f, vl);                               \
        auto minus_one = vfmv_v_f_f32m##LMUL(-1.f, vl);                        \
        auto p0 = vfmv_v_f_f32m##LMUL(0x1.55555ep-3, vl);                      \
        auto p1 = vfmv_v_f_f32m##LMUL(0x1.33261ap-4, vl);                      \
        auto p2 = vfmv_v_f_f32m##LMUL(0x1.70d7dcp-5, vl);                      \
        auto neg_mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);             \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto off = vfmerge_vfm_f32m##LMUL(neg_mask, zero, 0x1.921fb6p+1f, vl); \
        auto mul1 = vfmerge_vfm_f32m##LMUL(neg_mask, two, -2.f, vl);           \
        auto mul2 = vfmerge_vfm_f32m##LMUL(neg_mask, minus_one, 1.f, vl);      \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto le_half_mask = vmfle_vv_f32m##LMUL##_b##MLEN(x, half, vl);        \
        auto tmp = vmv_v_v_f32m##LMUL(x, vl);                                  \
        auto mul = vmerge_vvm_f32m##LMUL(le_half_mask, mul1, mul2, vl);        \
        tmp = vfnmsub_vv_f32m##LMUL(tmp, half, half, vl);                      \
        auto add =                                                             \
            vfmerge_vfm_f32m##LMUL(le_half_mask, off, 0x1.921fb6p+0f, vl);     \
        auto z2 = vfmul_vv_f32m##LMUL##_m(le_half_mask, tmp, v, v, vl);        \
        /* acos(|x|) = pi/2 - sign(x) * Q(|x|), for  |x| < 0.5                 \
                = 2 Q(|x|)               , for  0.5 < x < 1.0                  \
                = pi - 2 Q(|x|)          , for -1.0 < x < -0.5.  */            \
        auto y1 = vfmv_v_f_f32m##LMUL(0x1.3af7d8p-5, vl);                      \
        auto y2 = vfmv_v_f_f32m##LMUL(0x1.b059dp-6, vl);                       \
        tmp = vfsqrt_v_f32m##LMUL(z2, vl);                                     \
        auto z4 = vfmul_vv_f32m##LMUL(z2, z2, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, z4, p2, vl);                             \
        y2 = vfmadd_vv_f32m##LMUL(y2, z4, p1, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, z4, p0, vl);                             \
        auto z = vmerge_vvm_f32m##LMUL(le_half_mask, tmp, x, vl);              \
        y1 = vfmacc_vv_f32m##LMUL(y1, y2, z2, vl);                             \
        mul = vfmul_vv_f32m##LMUL(mul, z, vl);                                 \
        y1 = vfmadd_vv_f32m##LMUL(y1, z2, one, vl);                            \
        return vfmadd_vv_f32m##LMUL(y1, mul, add, vl);                         \
    }
#endif
IMPL_RVV_WITH_LMULS(ACOS_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, acos)

// acosh
// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
#define ACOSH_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t acosh_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto sub =                                                             \
            vfsub_vf_f32m##LMUL(vfmul_vv_f32m##LMUL(v, v, vl), 1.f, vl);       \
        auto sqrt = vfrec7_v_f32m##LMUL(vfrsqrt7_v_f32m##LMUL(sub, vl), vl);   \
        return log_ps(vfadd_vv_f32m##LMUL(v, sqrt, vl), vl);                   \
    }

IMPL_RVV_WITH_LMULS(ACOSH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, acosh)

// asin
#if 0
// porting from https://developer.download.nvidia.cn/cg/asin.html
#define ASIN_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t asin_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto c2 = vfmv_v_f_f32m##LMUL(0.0742610f, vl);                         \
        auto c3 = vfmv_v_f_f32m##LMUL(-0.2121144f, vl);                        \
        auto c4 = vfmv_v_f_f32m##LMUL(1.5707288f, vl);                         \
        auto c5 = vfmv_v_f_f32m##LMUL(3.14159265358979f * 0.5f, vl);           \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);                 \
        auto sroot =                                                           \
            vfsqrt_v_f32m##LMUL(vfrsub_vf_f32m##LMUL(x, 1.f, vl), vl);         \
        auto ret = vmv_v_v_f32m##LMUL(x, vl);                                  \
        ret = vfmadd_vf_f32m##LMUL(ret, -0.0187293f, c2, vl);                  \
        ret = vfmadd_vv_f32m##LMUL(ret, x, c3, vl);                            \
        ret = vfmadd_vv_f32m##LMUL(ret, x, c4, vl);                            \
        ret = vfnmsub_vv_f32m##LMUL(ret, sroot, c5, vl);                       \
        return vfneg_v_f32m##LMUL##_m(mask, ret, ret, vl);                     \
    }
#else
// from glibc 2.40: sysdeps/aarch64/fpu/asinf_advsimd.c
#define ASIN_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t asin_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto half = vfmv_v_f_f32m##LMUL(0.5f, vl);                             \
        auto one = vfmv_v_f_f32m##LMUL(1.f, vl);                               \
        auto minus_two = vfmv_v_f_f32m##LMUL(-2.f, vl);                        \
        auto pi_over_2f = vfmv_v_f_f32m##LMUL(0x1.921fb6p+0f, vl);             \
        auto p0 = vfmv_v_f_f32m##LMUL(0x1.55555ep-3, vl);                      \
        auto p1 = vfmv_v_f_f32m##LMUL(0x1.33261ap-4, vl);                      \
        auto p2 = vfmv_v_f_f32m##LMUL(0x1.70d7dcp-5, vl);                      \
        auto neg_mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);             \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto mul1 = vfmerge_vfm_f32m##LMUL(neg_mask, one, -1.f, vl);           \
                                                                               \
        /* Evaluate polynomial Q(x) = z + z * z2 * P(z2) with                  \
            z2 = x ^ 2         and z = |x|     , if |x| < 0.5                  \
            z2 = (1 - |x|) / 2 and z = sqrt(z2), if |x| >= 0.5.  */            \
        auto lt_half_mask = vmflt_vv_f32m##LMUL##_b##MLEN(x, half, vl);        \
        auto tmp = vmv_v_v_f32m##LMUL(x, vl);                                  \
        auto mul2 = vfmerge_vfm_f32m##LMUL(lt_half_mask, minus_two, 1.f, vl);  \
        tmp = vfnmsub_vv_f32m##LMUL(tmp, half, half, vl);                      \
        auto add = vfmerge_vfm_f32m##LMUL(lt_half_mask, pi_over_2f, 0.f, vl);  \
        auto z2 = vfmul_vv_f32m##LMUL##_m(lt_half_mask, tmp, v, v, vl);        \
        /* asin(|x|) = Q(|x|),        for |x| < 0.5                            \
                = pi / 2 - 2 Q(|x|) , for |x| >= 0.5.  */                      \
        auto y1 = vfmv_v_f_f32m##LMUL(0x1.3af7d8p-5, vl);                      \
        auto y2 = vfmv_v_f_f32m##LMUL(0x1.b059dp-6, vl);                       \
        auto z4 = vfmul_vv_f32m##LMUL(z2, z2, vl);                             \
        tmp = vfsqrt_v_f32m##LMUL(z2, vl);                                     \
        y1 = vfmadd_vv_f32m##LMUL(y1, z4, p2, vl);                             \
        y2 = vfmadd_vv_f32m##LMUL(y2, z4, p1, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, z4, p0, vl);                             \
        auto z = vmerge_vvm_f32m##LMUL(lt_half_mask, tmp, x, vl);              \
        y1 = vfmacc_vv_f32m##LMUL(y1, y2, z2, vl);                             \
        z2 = vfmul_vv_f32m##LMUL(z2, z, vl);                                   \
        y1 = vfmadd_vv_f32m##LMUL(y1, z2, z, vl);                              \
        y1 = vfmadd_vv_f32m##LMUL(y1, mul2, add, vl);                          \
        return vfmul_vv_f32m##LMUL(y1, mul1, vl);                              \
    }
#endif
IMPL_RVV_WITH_LMULS(ASIN_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, asin)

// asinh
// asinh(v) = ln(v + sqrt(v^2 + 1)), -inf < x < +inf
#if 0
#define ASINH_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t asinh_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto x = vfsgnj_vf_f32##m##LMUL(v, 1.f, vl);                           \
        auto sum =                                                             \
            vfadd_vf_f32m##LMUL(vfmul_vv_f32m##LMUL(v, v, vl), 1.f, vl);       \
        auto sqrt = vfrec7_v_f32m##LMUL(vfrsqrt7_v_f32m##LMUL(sum, vl), vl);   \
        auto ret = log_ps(vfadd_vv_f32m##LMUL(x, sqrt, vl), vl);               \
        return vfsgnj_vv_f32##m##LMUL(ret, v, vl);                             \
    }
#else
#define ASINH_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t asinh_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto sum = vfmv_v_f_f32m##LMUL(1.f, vl);                               \
        auto x = vfsgnj_vf_f32##m##LMUL(v, 1.f, vl);                           \
        sum = vfmacc_vv_f32m##LMUL(sum, v, v, vl);                             \
        auto sqrt = vfrec7_v_f32m##LMUL(vfrsqrt7_v_f32m##LMUL(sum, vl), vl);   \
        auto ret = log_ps(vfadd_vv_f32m##LMUL(x, sqrt, vl), vl);               \
        return vfsgnj_vv_f32##m##LMUL(ret, v, vl);                             \
    }
#endif
IMPL_RVV_WITH_LMULS(ASINH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, asinh)

// ceil
#define CEIL_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t ceil_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto vi = vfcvt_x_f_v_i32m##LMUL(v, vl);                               \
        auto vf = vfcvt_f_x_v_f32m##LMUL(vi, vl);                              \
        auto mask = vmflt_vv_f32m##LMUL##_b##MLEN(vf, v, vl);                  \
        return vfadd_vf_f32m##LMUL##_m(mask, vf, vf, 1.f, vl);                 \
    }

IMPL_RVV_WITH_LMULS(CEIL_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, ceil)

// cos
// from glibc 2.40: sysdeps/aarch64/fpu/cosf_advsimd.c
#define COS_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t cos_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        auto n = vfmv_v_f_f32m##LMUL(0x1.45f306p-2f, vl);                      \
        auto half = vfmv_v_f_f32m##LMUL(0.5f, vl);                             \
        auto c0 = vfmv_v_f_f32m##LMUL(-0x1.555548p-3f, vl);                    \
        auto c2 = vfmv_v_f_f32m##LMUL(-0x1.9f42eap-13f, vl);                   \
                                                                               \
        /*  n = rint((|x|+pi/2)/pi) - 0.5. */                                  \
        auto r = vfabs_v_f32m##LMUL(v, vl);                                    \
        n = vfmadd_vv_f32m##LMUL(n, r, half, vl);                              \
        auto ni = vfcvt_x_f_v_i32m##LMUL(n, vl);                               \
        n = vfcvt_f_x_v_f32m##LMUL(ni, vl);                                    \
        auto odd = vadd_vx_i32m##LMUL(ni, 0x1.8p+23, vl);                      \
        n = vfsub_vf_f32m##LMUL(n, 0.5f, vl);                                  \
        odd = vsll_vx_i32##m##LMUL(odd, 31, vl);                               \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = vfnmsac_vf_f32m##LMUL(r, 0x1.921fb6p+1f, n, vl);                   \
        r = vfnmsac_vf_f32m##LMUL(r, -0x1.777a5cp-24f, n, vl);                 \
        r = vfnmsac_vf_f32m##LMUL(r, -0x1.ee59dap-49f, n, vl);                 \
                                                                               \
        /* y = sin(r).  */                                                     \
        auto r2 = vfmul_vv_f32m##LMUL(r, r, vl);                               \
        auto y1 = vfmv_v_f_f32m##LMUL(0x1.5b2e76p-19f, vl);                    \
        auto y2 = vfmv_v_f_f32m##LMUL(0x1.110df4p-7f, vl);                     \
        y1 = vfmadd_vv_f32m##LMUL(y1, r2, c2, vl);                             \
        y2 = vfmadd_vv_f32m##LMUL(y2, r2, c0, vl);                             \
        auto r4 = vfmul_vv_f32m##LMUL(r2, r2, vl);                             \
        auto r3 = vfmul_vv_f32m##LMUL(r2, r, vl);                              \
        y1 = vfmadd_vv_f32m##LMUL(y1, r4, y2, vl);                             \
        y1 = vfmadd_vv_f32m##LMUL(y1, r3, r, vl);                              \
        auto tmp = vreinterpret_v_f32m##LMUL##_i32m##LMUL(y1);                 \
        tmp = vxor_vv_i32m##LMUL(tmp, odd, vl);                                \
        return vreinterpret_v_i32m##LMUL##_f32m##LMUL(tmp);                    \
    }

IMPL_RVV_WITH_LMULS(COS_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, cos)

// cosh(v) = (exp(v) + exp(-v)) / 2
#if 0
#define COSH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t cosh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = exp_ps(vfneg_v_f32m##LMUL(v, vl), vl);                        \
        auto sum = vfadd_vv_f32m##LMUL(a, b, vl);                              \
        return vfdiv_vf_f32m##LMUL(sum, 2.f, vl);                              \
    }
#else
#if 0
// max_ulp_error = 90164
#define COSH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t cosh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = vfrec7_v_f32m##LMUL(a, vl);                                   \
        auto sum = vfadd_vv_f32m##LMUL(a, b, vl);                              \
        return vfmul_vf_f32m##LMUL(sum, 0.5f, vl);                             \
    }
#else
#define COSH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t cosh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = vfrdiv_vf_f32m##LMUL(a, 1.f, vl);                             \
        auto sum = vfadd_vv_f32m##LMUL(a, b, vl);                              \
        return vfmul_vf_f32m##LMUL(sum, 0.5f, vl);                             \
    }
#endif
#endif
IMPL_RVV_WITH_LMULS(COSH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, cosh)

// exp
#define EXP_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t exp_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return exp_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(EXP_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, exp)

// floor
#define FLOOR_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t floor_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto vi = vfcvt_x_f_v_i32m##LMUL(v, vl);                               \
        auto vf = vfcvt_f_x_v_f32m##LMUL(vi, vl);                              \
        auto mask = vmfgt_vv_f32m##LMUL##_b##MLEN(vf, v, vl);                  \
        return vfsub_vf_f32m##LMUL##_m(mask, vf, vf, 1.f, vl);                 \
    }

IMPL_RVV_WITH_LMULS(FLOOR_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, floor)

// log
#define LOG_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t log_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return log_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(LOG_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, log)

// neg
#define NEG_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t neg_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return vfneg_v_f32m##LMUL(v, vl);                                      \
    }

IMPL_RVV_WITH_LMULS(NEG_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, neg)

// round
#define ROUND_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t round_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfcvt_f_x_v_f32m##LMUL(vfcvt_x_f_v_i32m##LMUL(v, vl), vl);      \
    }

IMPL_RVV_WITH_LMULS(ROUND_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, round)

// rsqrt
#if 0
// max_ulp_error = 0
#define RSQRT_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t rsqrt_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfrdiv_vf_f32m##LMUL(vfsqrt_v_f32m##LMUL(v, vl), 1.f, vl);      \
    }
#else
#if 0
// max_ulp_error = 88880
#define RSQRT_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t rsqrt_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfrsqrt7_v_f32m##LMUL(v, vl);                                   \
    }
#else
#define RSQRT_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t rsqrt_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto one_point_five = vfmv_v_f_f32m##LMUL(1.5f, vl);                   \
                                                                               \
        auto ux = vreinterpret_v_f32m##LMUL##_u32m##LMUL(v);                   \
        ux = vsrl_vx_u32m##LMUL(ux, 1, vl);                                    \
        ux = vrsub_vx_u32m##LMUL(ux, 0x5f375a86, vl);                          \
        auto y = vreinterpret_v_u32m##LMUL##_f32m##LMUL(ux);                   \
                                                                               \
        auto y2 = vfmul_vv_f32m##LMUL(y, y, vl);                               \
        auto x = vfmul_vf_f32m##LMUL(v, -0.5f, vl);                            \
        y2 = vfmadd_vv_f32m##LMUL(y2, x, one_point_five, vl);                  \
        y = vfmul_vv_f32m##LMUL(y, y2, vl);                                    \
                                                                               \
        y2 = vfmul_vv_f32m##LMUL(y, y, vl);                                    \
        y2 = vfmadd_vv_f32m##LMUL(y2, x, one_point_five, vl);                  \
        y = vfmul_vv_f32m##LMUL(y, y2, vl);                                    \
                                                                               \
        y2 = vfmul_vv_f32m##LMUL(y, y, vl);                                    \
        y2 = vfmadd_vv_f32m##LMUL(y2, x, one_point_five, vl);                  \
        y = vfmul_vv_f32m##LMUL(y, y2, vl);                                    \
                                                                               \
        y2 = vfmul_vv_f32m##LMUL(y, y, vl);                                    \
        y2 = vfmadd_vv_f32m##LMUL(y2, x, one_point_five, vl);                  \
        y = vfmul_vv_f32m##LMUL(y, y2, vl);                                    \
        return y;                                                              \
    }
#endif
#endif

IMPL_RVV_WITH_LMULS(RSQRT_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, rsqrt)

// sign
#define SIGN_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t sign_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto ret = vfmv_v_f_f32m##LMUL(0.f, vl);                               \
        auto gt_mask = vmfgt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);              \
        ret = vfmerge_vfm_f32m##LMUL(gt_mask, ret, 1.f, vl);                   \
        auto lt_mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);              \
        return vfmerge_vfm_f32m##LMUL(lt_mask, ret, -1.f, vl);                 \
    }

IMPL_RVV_WITH_LMULS(SIGN_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, sign)

// sin
// from glibc 2.40: sysdeps/aarch64/fpu/sinf_advsimd.c
#define SIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sin_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        auto c0 = vfmv_v_f_f32m##LMUL(-0x1.555548p-3f, vl);                    \
        auto c2 = vfmv_v_f_f32m##LMUL(-0x1.9f42eap-13f, vl);                   \
                                                                               \
        /* n = rint(|x|/pi) */                                                 \
        auto r = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto n = vfmul_vf_f32m##LMUL(r, 0x1.45f306p-2f, vl);                   \
        auto sign =                                                            \
            vxor_vv_i32m##LMUL(vreinterpret_v_f32m##LMUL##_i32m##LMUL(v),      \
                               vreinterpret_v_f32m##LMUL##_i32m##LMUL(r), vl); \
        auto ni = vfcvt_x_f_v_i32m##LMUL(n, vl);                               \
        n = vfcvt_f_x_v_f32m##LMUL(ni, vl);                                    \
        auto odd = vadd_vx_i32m##LMUL(ni, 0x1.8p+23, vl);                      \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = vfnmsac_vf_f32m##LMUL(r, 0x1.921fb6p+1f, n, vl);                   \
        odd = vsll_vx_i32##m##LMUL(odd, 31, vl);                               \
        r = vfnmsac_vf_f32m##LMUL(r, -0x1.777a5cp-24f, n, vl);                 \
        r = vfnmsac_vf_f32m##LMUL(r, -0x1.ee59dap-49f, n, vl);                 \
                                                                               \
        /* y = sin(r).  */                                                     \
        auto r2 = vfmul_vv_f32m##LMUL(r, r, vl);                               \
        auto y1 = vfmv_v_f_f32m##LMUL(0x1.5b2e76p-19f, vl);                    \
        auto y2 = vfmv_v_f_f32m##LMUL(0x1.110df4p-7f, vl);                     \
        y1 = vfmadd_vv_f32m##LMUL(y1, r2, c2, vl);                             \
        y2 = vfmadd_vv_f32m##LMUL(y2, r2, c0, vl);                             \
        auto r4 = vfmul_vv_f32m##LMUL(r2, r2, vl);                             \
        auto r3 = vfmul_vv_f32m##LMUL(r2, r, vl);                              \
        y1 = vfmadd_vv_f32m##LMUL(y1, r4, y2, vl);                             \
        sign = vxor_vv_i32m##LMUL(sign, odd, vl);                              \
        y1 = vfmadd_vv_f32m##LMUL(y1, r3, r, vl);                              \
        auto tmp = vreinterpret_v_f32m##LMUL##_i32m##LMUL(y1);                 \
        tmp = vxor_vv_i32m##LMUL(tmp, sign, vl);                               \
        return vreinterpret_v_i32m##LMUL##_f32m##LMUL(tmp);                    \
    }

IMPL_RVV_WITH_LMULS(SIN_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, sin)

// sinh(v) = (exp(v) - exp(-v)) / 2
#define SINH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t sinh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = vfrec7_v_f32m##LMUL(a, vl);                                   \
        return vfmul_vf_f32m##LMUL(vfsub_vv_f32m##LMUL(a, b, vl), 0.5f, vl);   \
    }

IMPL_RVV_WITH_LMULS(SINH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, sinh)

// sqrt
#define SQRT_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t sqrt_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        return vfsqrt_v_f32m##LMUL(v, vl);                                     \
    }

IMPL_RVV_WITH_LMULS(SQRT_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, sqrt)

// square
#define SQUARE_FLOAT32(LMUL, MLEN)                                             \
    inline vfloat32m##LMUL##_t square_float32(const vfloat32m##LMUL##_t &v,    \
                                              const size_t vl) {               \
        return vfmul_vv_f32m##LMUL(v, v, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(SQUARE_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, square)

// tanh
#define TANH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t tanh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        return tanh_ps(v, vl);                                                 \
    }

IMPL_RVV_WITH_LMULS(TANH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, tanh)

// binary
#define RVV_BINARY_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel)        \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,                     \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v1,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v2)      \
            const noexcept {                                                   \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input1 = vle##sew##_v_##dtype_prefix##sew##m##lmul(           \
                p1, NTT_VL(vlen, sew, lmul));                                  \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(           \
                p2, NTT_VL(vlen, sew, lmul));                                  \
            auto output = kernel(input1, input2, NTT_VL(vlen, sew, lmul));     \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
                                                                               \
            return vout;                                                       \
        }                                                                      \
    };
#if 0
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>, dtype> {            \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v,       \
                   const dtype &s) const noexcept {                            \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p = reinterpret_cast<const dtype *>(v.elements().data());     \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input = vle##sew##_v_##dtype_prefix##sew##m##lmul(            \
                p, NTT_VL(vlen, sew, lmul));                                   \
            auto output = kernel(input, s, NTT_VL(vlen, sew, lmul));           \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
                                                                               \
            return vout;                                                       \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <>                                                                \
    struct op<dtype, ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {            \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const dtype &s,                                             \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v)       \
            const noexcept {                                                   \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p = reinterpret_cast<const dtype *>(v.elements().data());     \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input = vle##sew##_v_##dtype_prefix##sew##m##lmul(            \
                p, NTT_VL(vlen, sew, lmul));                                   \
            auto output = kernel(input, s, NTT_VL(vlen, sew, lmul));           \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
                                                                               \
            return vout;                                                       \
        }                                                                      \
    };
#endif
// binary with float
#define REGISTER_RVV_BINARY_OP_FLOAT32(OP, vlen)                               \
    RVV_BINARY_OP(OP, float, f, vlen, 32, 1, OP##_float32)                     \
    RVV_BINARY_OP(OP, float, f, vlen, 32, 2, OP##_float32)                     \
    RVV_BINARY_OP(OP, float, f, vlen, 32, 4, OP##_float32)                     \
    RVV_BINARY_OP(OP, float, f, vlen, 32, 8, OP##_float32)

// binary with int32_t
#define REGISTER_RVV_BINARY_OP_INT32(OP, vlen)                                 \
    RVV_BINARY_OP(OP, int32_t, i, vlen, 32, 1, OP##_int32)                     \
    RVV_BINARY_OP(OP, int32_t, i, vlen, 32, 2, OP##_int32)                     \
    RVV_BINARY_OP(OP, int32_t, i, vlen, 32, 4, OP##_int32)                     \
    RVV_BINARY_OP(OP, int32_t, i, vlen, 32, 8, OP##_int32)

// add
#define ADD_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t add_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfadd_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t add_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfadd_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t add_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfadd_vf_f32m##LMUL(v, s, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(ADD_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, add)

// sub
#define SUB_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sub_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfsub_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t sub_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfsub_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t sub_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfrsub_vf_f32m##LMUL(v, s, vl);                                 \
    }

IMPL_RVV_WITH_LMULS(SUB_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, sub)

// mul
#define MUL_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t mul_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmul_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mul_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfmul_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mul_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfmul_vf_f32m##LMUL(v, s, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(MUL_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, mul)

// div
#define DIV_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t div_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfdiv_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t div_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfdiv_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t div_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfrdiv_vf_f32m##LMUL(v, s, vl);                                 \
    }

IMPL_RVV_WITH_LMULS(DIV_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, div)

// mod
#define MOD_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t mod_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(vfdiv_vv_f32m##LMUL(v1, v2, vl), vl),   \
            vl);                                                               \
        return vfnmsub_vv_f32m##LMUL(quotient, v2, v1, vl);                    \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mod_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(vfdiv_vf_f32m##LMUL(v, s, vl), vl),     \
            vl);                                                               \
        return vfnmsub_vf_f32m##LMUL(quotient, s, v, vl);                      \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mod_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v2, const size_t vl) {      \
        auto v1 = vfmv_v_f_f32m##LMUL(s, vl);                                  \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(vfrdiv_vf_f32m##LMUL(v2, s, vl), vl),   \
            vl);                                                               \
        return vfnmsub_vv_f32m##LMUL(quotient, v2, v1, vl);                    \
    }

IMPL_RVV_WITH_LMULS(MOD_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, mod)

// min
#define MIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t min_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmin_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t min_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfmin_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t min_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfmin_vf_f32m##LMUL(v, s, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(MIN_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, min)

// max
#define MAX_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t max_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmax_vv_f32m##LMUL(v1, v2, vl);                                \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t max_float32(const vfloat32m##LMUL##_t &v,       \
                                           const float &s, const size_t vl) {  \
        return vfmax_vf_f32m##LMUL(v, s, vl);                                  \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t max_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v, const size_t vl) {       \
        return vfmax_vf_f32m##LMUL(v, s, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(MAX_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, max)

// pow
#define POW_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t pow_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return pow_ps(v1, v2, vl);                                             \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t pow_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const float &s, const size_t vl) {  \
        auto v2 = vfmv_v_f_f32m##LMUL(s, vl);                                  \
        return pow_ps(v1, v2, vl);                                             \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t pow_float32(                                    \
        const float &s, const vfloat32m##LMUL##_t &v2, const size_t vl) {      \
        auto v1 = vfmv_v_f_f32m##LMUL(s, vl);                                  \
        return pow_ps(v1, v2, vl);                                             \
    }

IMPL_RVV_WITH_LMULS(POW_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, pow)

// floor_mod
#define FLOOR_MOD_INT32(LMUL, MLEN)                                            \
    inline vint32m##LMUL##_t floor_mod_int32(const vint32m##LMUL##_t &v1,      \
                                             const vint32m##LMUL##_t &v2,      \
                                             const size_t vl) {                \
        auto remainder = vrem_vv_i32m##LMUL(v1, v2, vl);                       \
        auto sign1 = vsra_vx_i32m##LMUL(v1, 31, vl);                           \
        auto sign2 = vsra_vx_i32m##LMUL(v2, 31, vl);                           \
        auto cond1 = vmsne_vx_i32m##LMUL##_b##MLEN(remainder, 0, vl);          \
        auto cond2 = vmsne_vv_i32m##LMUL##_b##MLEN(sign1, sign2, vl);          \
        cond1 = vmand_mm_b##MLEN(cond1, cond2, vl);                            \
        return vadd_vv_i32m##LMUL##_m(cond1, remainder, remainder, v2, vl);    \
    }                                                                          \
                                                                               \
    inline vint32m##LMUL##_t floor_mod_int32(                                  \
        const vint32m##LMUL##_t &v1, const int32_t &s, const size_t vl) {      \
        auto remainder = vrem_vx_i32m##LMUL(v1, s, vl);                        \
        auto sign1 = vsra_vx_i32m##LMUL(v1, 31, vl);                           \
        auto sign2 = s >> 31;                                                  \
        auto cond1 = vmsne_vx_i32m##LMUL##_b##MLEN(remainder, 0, vl);          \
        auto cond2 = vmsne_vx_i32m##LMUL##_b##MLEN(sign1, sign2, vl);          \
        cond1 = vmand_mm_b##MLEN(cond1, cond2, vl);                            \
        return vadd_vx_i32m##LMUL##_m(cond1, remainder, remainder, s, vl);     \
    }                                                                          \
                                                                               \
    inline vint32m##LMUL##_t floor_mod_int32(                                  \
        const int32_t &s, const vint32m##LMUL##_t &v2, const size_t vl) {      \
        auto v1 = vmv_v_x_i32m##LMUL(s, vl);                                   \
        auto remainder = vrem_vv_i32m##LMUL(v1, v2, vl);                       \
        auto sign1 = s >> 31;                                                  \
        auto sign2 = vsra_vx_i32m##LMUL(v2, 31, vl);                           \
        auto cond1 = vmsne_vx_i32m##LMUL##_b##MLEN(remainder, 0, vl);          \
        auto cond2 = vmsne_vx_i32m##LMUL##_b##MLEN(sign2, sign1, vl);          \
        cond1 = vmand_mm_b##MLEN(cond1, cond2, vl);                            \
        return vadd_vv_i32m##LMUL##_m(cond1, remainder, remainder, v2, vl);    \
    }

IMPL_RVV_WITH_LMULS(FLOOR_MOD_INT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_INT32, floor_mod)

// swish
#define SWISH_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t swish_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        auto a = exp_ps(vfneg_v_f32m##LMUL(v, vl), vl);                        \
        auto b = vfrec7_v_f32m##LMUL(vfadd_vf_f32m##LMUL(a, 1.f, vl), vl);     \
        return vfmul_vv_f32m##LMUL(v, b, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(SWISH_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, swish)

#define RVV_SWISHB_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel)        \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>, dtype> {            \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v,       \
                   dtype beta) const noexcept {                                \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto pin = reinterpret_cast<const dtype *>(v.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input = vle##sew##_v_##dtype_prefix##sew##m##lmul(            \
                pin, NTT_VL(vlen, sew, lmul));                                 \
            auto output = kernel(input, NTT_VL(vlen, sew, lmul), beta);        \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
            return vout;                                                       \
        }                                                                      \
    };

#define REGISTER_RVV_SWISHB_OP_FLOAT32(OP, vlen)                               \
    RVV_SWISHB_OP(OP, float, f, vlen, 32, 1, OP##_float32)                     \
    RVV_SWISHB_OP(OP, float, f, vlen, 32, 2, OP##_float32)                     \
    RVV_SWISHB_OP(OP, float, f, vlen, 32, 4, OP##_float32)                     \
    RVV_SWISHB_OP(OP, float, f, vlen, 32, 8, OP##_float32)

// swishb
#define SWISHB_FLOAT32(LMUL, MLEN)                                             \
    inline vfloat32m##LMUL##_t swishb_float32(const vfloat32m##LMUL##_t &v,    \
                                              const size_t vl, float beta) {   \
        return swish_op(v, vl, beta);                                          \
    }

IMPL_RVV_WITH_LMULS(SWISHB_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_SWISHB_OP_FLOAT32, swishb)

// outer product
#define RVV_OUTER_PRODUCT_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel) \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,                     \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        fixed_tensor<dtype, NTT_VL(vlen, sew, lmul), NTT_VL(vlen, sew, lmul)>  \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v1,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v2)      \
            const noexcept {                                                   \
            constexpr size_t vl = NTT_VL(vlen, sew, lmul);                     \
            fixed_tensor<dtype, vl, vl> vout;                                  \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p2, vl);   \
            if (vl == 4) {                                                     \
                auto out0 = vfmul_vf_f32m##lmul(input2, p1[0], vl);            \
                auto out1 = vfmul_vf_f32m##lmul(input2, p1[1], vl);            \
                auto out2 = vfmul_vf_f32m##lmul(input2, p1[2], vl);            \
                auto out3 = vfmul_vf_f32m##lmul(input2, p1[3], vl);            \
                vse##sew##_v_##dtype_prefix##sew##m##lmul(pout, out0, vl);     \
                vse##sew##_v_##dtype_prefix##sew##m##lmul(pout + vl, out1,     \
                                                          vl);                 \
                vse##sew##_v_##dtype_prefix##sew##m##lmul(pout + 2 * vl, out2, \
                                                          vl);                 \
                vse##sew##_v_##dtype_prefix##sew##m##lmul(pout + 3 * vl, out3, \
                                                          vl);                 \
            } else {                                                           \
                for (size_t i = 0; i < vl; i++) {                              \
                    auto output = vfmul_vf_f32m##lmul(input2, p1[i], vl);      \
                    vse##sew##_v_##dtype_prefix##sew##m##lmul(pout, output,    \
                                                              vl);             \
                    pout += vl;                                                \
                }                                                              \
            }                                                                  \
            return vout;                                                       \
        }                                                                      \
    };

// outer product with float
#define REGISTER_RVV_OUTER_PRODUCT_OP_FLOAT32(OP, vlen)                        \
    RVV_OUTER_PRODUCT_OP(OP, float, f, vlen, 32, 1, OP##_float32)              \
    RVV_OUTER_PRODUCT_OP(OP, float, f, vlen, 32, 2, OP##_float32)              \
    RVV_OUTER_PRODUCT_OP(OP, float, f, vlen, 32, 4, OP##_float32)              \
    RVV_OUTER_PRODUCT_OP(OP, float, f, vlen, 32, 8, OP##_float32)

REGISTER_RVV_WITH_VLENS(REGISTER_RVV_OUTER_PRODUCT_OP_FLOAT32, outer_product)

// inner product
#define RVV_INNER_PRODUCT_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel) \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,                     \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        dtype                                                                  \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v1,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v2)      \
            const noexcept {                                                   \
            constexpr size_t vl = NTT_VL(vlen, sew, lmul);                     \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto input1 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p1, vl);   \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p2, vl);   \
            auto zero = vfmv_v_f_f32m1(0, vl);                                 \
            auto dest = vfmv_v_f_f32m1(0, vl);                                 \
            auto mul = vfmul_vv_f32m##lmul(input1, input2, vl);                \
            return vfmv_f_s_f32m1_f32(                                         \
                vfredusum_vs_f32m##lmul##_f32m1(dest, mul, zero, vl));         \
        }                                                                      \
    };

// inner product with float
#define REGISTER_RVV_INNER_PRODUCT_OP_FLOAT32(OP, vlen)                        \
    RVV_INNER_PRODUCT_OP(OP, float, f, vlen, 32, 1, OP##_float32)              \
    RVV_INNER_PRODUCT_OP(OP, float, f, vlen, 32, 2, OP##_float32)              \
    RVV_INNER_PRODUCT_OP(OP, float, f, vlen, 32, 4, OP##_float32)              \
    RVV_INNER_PRODUCT_OP(OP, float, f, vlen, 32, 8, OP##_float32)

REGISTER_RVV_WITH_VLENS(REGISTER_RVV_INNER_PRODUCT_OP_FLOAT32, inner_product)

// mul_add
#define RVV_MUL_ADD_OP(op, dtype, dtype_prefix, vlen, sew, lmul, kernel)       \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,                     \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,                     \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v1,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v2,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v3)      \
            const noexcept {                                                   \
            constexpr size_t vl = NTT_VL(vlen, sew, lmul);                     \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto p3 = reinterpret_cast<const dtype *>(v3.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input1 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p1, vl);   \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p2, vl);   \
            auto input3 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p3, vl);   \
            auto output = kernel(input1, input2, input3, vl);                  \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
            return vout;                                                       \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>, dtype,              \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v1,      \
                   const dtype &s2,                                            \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v3)      \
            const noexcept {                                                   \
            constexpr size_t vl = NTT_VL(vlen, sew, lmul);                     \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p3 = reinterpret_cast<const dtype *>(v3.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input1 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p1, vl);   \
            auto input3 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p3, vl);   \
            auto output = kernel(input1, s2, input3, vl);                      \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
            return vout;                                                       \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <>                                                                \
    struct op<dtype, ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>,              \
              ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>> {                   \
        ntt::vector<dtype, NTT_VL(vlen, sew, lmul)>                            \
        operator()(const dtype &s1,                                            \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v2,      \
                   const ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> &v3)      \
            const noexcept {                                                   \
            constexpr size_t vl = NTT_VL(vlen, sew, lmul);                     \
            ntt::vector<dtype, NTT_VL(vlen, sew, lmul)> vout;                  \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto p3 = reinterpret_cast<const dtype *>(v3.elements().data());   \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p2, vl);   \
            auto input3 = vle##sew##_v_##dtype_prefix##sew##m##lmul(p3, vl);   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto output = kernel(s1, input2, input3, vl);                      \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(                         \
                pout, output, NTT_VL(vlen, sew, lmul));                        \
            return vout;                                                       \
        }                                                                      \
    };

#define MUL_ADD_FLOAT32(LMUL, MLEN)                                            \
    inline vfloat32m##LMUL##_t mul_add_float32(                                \
        const vfloat32m##LMUL##_t &v1, const vfloat32m##LMUL##_t &v2,          \
        const vfloat32m##LMUL##_t &v3, const size_t vl) {                      \
        return vfmadd_vv_f32m##LMUL(v1, v2, v3, vl);                           \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mul_add_float32(                                \
        const vfloat32m##LMUL##_t &v1, const float &s2,                        \
        const vfloat32m##LMUL##_t &v3, const size_t vl) {                      \
        return vfmadd_vf_f32m##LMUL(v1, s2, v3, vl);                           \
    }                                                                          \
                                                                               \
    inline vfloat32m##LMUL##_t mul_add_float32(                                \
        const float &s1, const vfloat32m##LMUL##_t &v2,                        \
        const vfloat32m##LMUL##_t &v3, const size_t vl) {                      \
        return vfmadd_vf_f32m##LMUL(v2, s1, v3, vl);                           \
    }

// mul_add with float
#define REGISTER_RVV_MUL_ADD_OP_FLOAT32(OP, vlen)                              \
    RVV_MUL_ADD_OP(OP, float, f, vlen, 32, 1, OP##_float32)                    \
    RVV_MUL_ADD_OP(OP, float, f, vlen, 32, 2, OP##_float32)                    \
    RVV_MUL_ADD_OP(OP, float, f, vlen, 32, 4, OP##_float32)                    \
    RVV_MUL_ADD_OP(OP, float, f, vlen, 32, 8, OP##_float32)

IMPL_RVV_WITH_LMULS(MUL_ADD_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_MUL_ADD_OP_FLOAT32, mul_add)

#endif
} // namespace nncase::ntt::ops
