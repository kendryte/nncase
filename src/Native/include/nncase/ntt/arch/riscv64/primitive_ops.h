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
// cos(x) = 1 - x^2/2! + x^4/4! - x^6/6!
#define COS_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t cos_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(                                        \
                vfmul_vf_f32m##LMUL(x, 1 / (2.f * M_PI), vl), vl),             \
            vl);                                                               \
        x = vfnmsub_vf_f32m##LMUL(quotient, 2.f * M_PI, x, vl);                \
        auto mask1 = vmfgt_vf_f32m##LMUL##_b##MLEN(x, M_PI, vl);               \
        x = vfsub_vf_f32m##LMUL##_m(mask1, x, x, M_PI, vl);                    \
        auto mask2 = vmfgt_vf_f32m##LMUL##_b##MLEN(x, M_PI / 2.f, vl);         \
        x = vfrsub_vf_f32m##LMUL##_m(mask2, x, x, M_PI, vl);                   \
        auto x2 = vfmul_vv_f32m##LMUL(x, x, vl);                               \
        auto ret = vfmv_v_f_f32m##LMUL(1 / 24.f, vl);                          \
        auto c1 = vfmv_v_f_f32m##LMUL(-1 / 2.f, vl);                           \
        auto c2 = vfmv_v_f_f32m##LMUL(1.f, vl);                                \
        ret = vfmacc_vf_f32m##LMUL(ret, -1 / 720.f, x2, vl);                   \
        ret = vfmadd_vv_f32m##LMUL(ret, x2, c1, vl);                           \
        ret = vfmadd_vv_f32m##LMUL(ret, x2, c2, vl);                           \
        auto mask = vmxor_mm_b##MLEN(mask1, mask2, vl);                        \
        return vfneg_v_f32m##LMUL##_m(mask, ret, ret, vl);                     \
    }

IMPL_RVV_WITH_LMULS(COS_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_UNARY_OP_FLOAT32, cos)

// cosh(v) = (exp(v) + exp(-v)) / 2
#define COSH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t cosh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto a = exp_ps(v, vl);                                                \
        auto b = vfrec7_v_f32m##LMUL(a, vl);                                   \
        return vfmul_vf_f32m##LMUL(vfadd_vv_f32m##LMUL(a, b, vl), 0.5f, vl);   \
    }

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
#define RSQRT_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t rsqrt_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfrsqrt7_v_f32m##LMUL(v, vl);                                   \
    }

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
// sin(x) = x - x^3/3! + x^5/5! - x^7/7!
#define SIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sin_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        auto mask1 = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);                \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(                                        \
                vfmul_vf_f32m##LMUL(x, 1 / (2.f * M_PI), vl), vl),             \
            vl);                                                               \
        x = vfnmsub_vf_f32m##LMUL(quotient, 2.f * M_PI, x, vl);                \
        auto mask2 = vmfgt_vf_f32m##LMUL##_b##MLEN(x, M_PI, vl);               \
        x = vfsub_vf_f32m##LMUL##_m(mask2, x, x, M_PI, vl);                    \
        auto mask3 = vmfgt_vf_f32m##LMUL##_b##MLEN(x, M_PI / 2.f, vl);         \
        x = vfrsub_vf_f32m##LMUL##_m(mask3, x, x, M_PI, vl);                   \
        auto x2 = vfmul_vv_f32m##LMUL(x, x, vl);                               \
        auto ret = vfmv_v_f_f32m##LMUL(1 / 120.f, vl);                         \
        auto c1 = vfmv_v_f_f32m##LMUL(-1 / 6.f, vl);                           \
        auto c2 = vfmv_v_f_f32m##LMUL(1.f, vl);                                \
        ret = vfmacc_vf_f32m##LMUL(ret, -1 / 5040.f, x2, vl);                  \
        ret = vfmadd_vv_f32m##LMUL(ret, x2, c1, vl);                           \
        ret = vfmadd_vv_f32m##LMUL(ret, x2, c2, vl);                           \
        ret = vfmul_vv_f32m##LMUL(x, ret, vl);                                 \
        auto mask = vmxor_mm_b##MLEN(mask1, mask2, vl);                        \
        return vfneg_v_f32m##LMUL##_m(mask, ret, ret, vl);                     \
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
    }

IMPL_RVV_WITH_LMULS(ADD_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, add)

// sub
#define SUB_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sub_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfsub_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(SUB_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, sub)

// mul
#define MUL_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t mul_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmul_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MUL_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, mul)

// div
#define DIV_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t div_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfdiv_vv_f32m##LMUL(v1, v2, vl);                                \
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
    }
IMPL_RVV_WITH_LMULS(MOD_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, mod)

// min
#define MIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t min_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmin_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MIN_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, min)

// max
#define MAX_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t max_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmax_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MAX_FLOAT32)
REGISTER_RVV_WITH_VLENS(REGISTER_RVV_BINARY_OP_FLOAT32, max)

// pow
#define POW_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t pow_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
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
        auto cond1 = vmsne_vx_i32m##LMUL##_b##MLEN(remainder, 0, vl);          \
        auto sign1 = vsra_vx_i32m##LMUL(v1, 31, vl);                           \
        auto sign2 = vsra_vx_i32m##LMUL(v2, 31, vl);                           \
        auto cond2 = vmsne_vv_i32m##LMUL##_b##MLEN(sign1, sign2, vl);          \
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

#endif
} // namespace nncase::ntt::ops