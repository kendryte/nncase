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

#define RVV_UNARY_OP(op, dtype, dtype_prefix, sew, lmul, kernel)               \
    template <> struct op<ntt::vector<dtype, NTT_VL(sew, lmul)>> {             \
        ntt::vector<dtype, NTT_VL(sew, lmul)> operator()(                      \
            const ntt::vector<dtype, NTT_VL(sew, lmul)> &v) const noexcept {   \
            ntt::vector<dtype, NTT_VL(sew, lmul)> vout;                        \
            auto pin = reinterpret_cast<const dtype *>(v.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input = vle##sew##_v_##dtype_prefix##sew##m##lmul(            \
                pin, NTT_VL(sew, lmul));                                       \
            auto output = kernel(input, NTT_VL(sew, lmul));                    \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(pout, output,            \
                                                      NTT_VL(sew, lmul));      \
            return vout;                                                       \
        }                                                                      \
    };

// unary with float
#define REGISTER_RVV_UNARY_OP_FLOAT32(OP)                                      \
    RVV_UNARY_OP(OP, float, f, 32, 1, OP##_float32)                            \
    RVV_UNARY_OP(OP, float, f, 32, 2, OP##_float32)                            \
    RVV_UNARY_OP(OP, float, f, 32, 4, OP##_float32)                            \
    RVV_UNARY_OP(OP, float, f, 32, 8, OP##_float32)

// abs
#define ABS_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t abs_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return vfabs_v_f32m##LMUL(v, vl);                                      \
    }

IMPL_RVV_WITH_LMULS(ABS_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(abs)

// acos
// porting from https://developer.download.nvidia.cn/cg/acos.html
#define ACOS_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t acos_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto ret = vfmul_vf_f32m##LMUL(x, -0.0187293f, vl);                    \
                                                                               \
        ret = vfadd_vf_f32m##LMUL(ret, 0.0742610f, vl);                        \
        ret = vfmul_vv_f32m##LMUL(ret, x, vl);                                 \
                                                                               \
        ret = vfsub_vf_f32m##LMUL(ret, 0.2121144f, vl);                        \
        ret = vfmul_vv_f32m##LMUL(ret, x, vl);                                 \
                                                                               \
        ret = vfadd_vf_f32m##LMUL(ret, 1.5707288f, vl);                        \
        ret = vfmul_vv_f32m##LMUL(                                             \
            ret, vfsqrt_v_f32m##LMUL(vfrsub_vf_f32m##LMUL(x, 1.f, vl), vl),    \
            vl);                                                               \
        auto mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);                 \
        ret = vfmul_vf_f32m##LMUL##_m(mask, ret, ret, -1.f, vl);               \
        return vfadd_vf_f32m##LMUL##_m(mask, ret, ret, 3.14159265358979f, vl); \
    }

IMPL_RVV_WITH_LMULS(ACOS_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(acos)

// asin
// porting from https://developer.download.nvidia.cn/cg/asin.html
#define ASIN_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t asin_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto x = vfabs_v_f32m##LMUL(v, vl);                                    \
        auto ret = vfmul_vf_f32m##LMUL(x, -0.0187293f, vl);                    \
                                                                               \
        ret = vfadd_vf_f32m##LMUL(ret, 0.0742610f, vl);                        \
        ret = vfmul_vv_f32m##LMUL(ret, x, vl);                                 \
                                                                               \
        ret = vfsub_vf_f32m##LMUL(ret, 0.2121144f, vl);                        \
        ret = vfmul_vv_f32m##LMUL(ret, x, vl);                                 \
                                                                               \
        ret = vfadd_vf_f32m##LMUL(ret, 1.5707288f, vl);                        \
        ret = vfrsub_vf_f32m##LMUL(                                            \
            vfmul_vv_f32m##LMUL(                                               \
                vfsqrt_v_f32m##LMUL(vfrsub_vf_f32m##LMUL(x, 1.f, vl), vl),     \
                ret, vl),                                                      \
            3.14159265358979f * 0.5f, vl);                                     \
        auto mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);                 \
        return vfmul_vf_f32m##LMUL##_m(mask, ret, ret, -1.f, vl);              \
    }

IMPL_RVV_WITH_LMULS(ASIN_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(asin)

// ceil
#define CEIL_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t ceil_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        vint32m##LMUL##_t vi = vfcvt_x_f_v_i32m##LMUL(v, vl);                  \
        auto _mask = vmflt_vv_f32m##LMUL##_b##MLEN(                            \
            vfcvt_f_x_v_f32m##LMUL(vi, vl), v, vl);                            \
        return vfcvt_f_x_v_f32m##LMUL(                                         \
            vadd_vx_i32m##LMUL##_m(_mask, vi, vi, 1, vl), vl);                 \
    }

IMPL_RVV_WITH_LMULS(CEIL_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(ceil)

// cos
#define COS_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t cos_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return cos_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(COS_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(cos)

// exp
#define EXP_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t exp_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return exp_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(EXP_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(exp)

// floor
#define FLOOR_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t floor_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        vint32m##LMUL##_t vi = vfcvt_x_f_v_i32m##LMUL(v, vl);                  \
        auto _mask = vmfgt_vv_f32m##LMUL##_b##MLEN(                            \
            vfcvt_f_x_v_f32m##LMUL(vi, vl), v, vl);                            \
        return vfcvt_f_x_v_f32m##LMUL(                                         \
            vsub_vx_i32m##LMUL##_m(_mask, vi, vi, 1, vl), vl);                 \
    }

IMPL_RVV_WITH_LMULS(FLOOR_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(floor)

// log
#define LOG_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t log_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return log_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(LOG_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(log)

// neg
#define NEG_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t neg_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return vfneg_v_f32m##LMUL(v, vl);                                      \
    }

IMPL_RVV_WITH_LMULS(NEG_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(neg)

// round
#define ROUND_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t round_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfcvt_f_x_v_f32m##LMUL(vfcvt_x_f_v_i32m##LMUL(v, vl), vl);      \
    }

IMPL_RVV_WITH_LMULS(ROUND_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(round)

// rsqrt
#define RSQRT_FLOAT32(LMUL, MLEN)                                              \
    inline vfloat32m##LMUL##_t rsqrt_float32(const vfloat32m##LMUL##_t &v,     \
                                             const size_t vl) {                \
        return vfrsqrt7_v_f32m##LMUL(v, vl);                                   \
    }

IMPL_RVV_WITH_LMULS(RSQRT_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(rsqrt)

// sign
#define SIGN_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t sign_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        auto zeros = vfmv_v_f_f32m##LMUL(0.f, vl);                             \
        auto gt_mask = vmfgt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);              \
        auto ret = vfadd_vf_f32m##LMUL##_m(gt_mask, zeros, zeros, 1.f, vl);    \
        auto lt_mask = vmflt_vf_f32m##LMUL##_b##MLEN(v, 0.f, vl);              \
        return vfsub_vf_f32m##LMUL##_m(lt_mask, ret, ret, 1.f, vl);            \
    }

IMPL_RVV_WITH_LMULS(SIGN_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(sign)

// sin
#define SIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sin_float32(const vfloat32m##LMUL##_t &v,       \
                                           const size_t vl) {                  \
        return sin_ps(v, vl);                                                  \
    }

IMPL_RVV_WITH_LMULS(SIN_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(sin)

// sqrt
#define SQRT_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t sqrt_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        return vfsqrt_v_f32m##LMUL(v, vl);                                     \
    }

IMPL_RVV_WITH_LMULS(SQRT_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(sqrt)

// square
#define SQUARE_FLOAT32(LMUL, MLEN)                                             \
    inline vfloat32m##LMUL##_t square_float32(const vfloat32m##LMUL##_t &v,    \
                                              const size_t vl) {               \
        return vfmul_vv_f32m##LMUL(v, v, vl);                                  \
    }

IMPL_RVV_WITH_LMULS(SQUARE_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(square)

// tanh
#define TANH_FLOAT32(LMUL, MLEN)                                               \
    inline vfloat32m##LMUL##_t tanh_float32(const vfloat32m##LMUL##_t &v,      \
                                            const size_t vl) {                 \
        return tanh_ps(v, vl);                                                 \
    }

IMPL_RVV_WITH_LMULS(TANH_FLOAT32)
REGISTER_RVV_UNARY_OP_FLOAT32(tanh)

// binary
#define RVV_BINARY_OP(op, dtype, dtype_prefix, sew, lmul, kernel)              \
    template <>                                                                \
    struct op<ntt::vector<dtype, NTT_VL(sew, lmul)>,                           \
              ntt::vector<dtype, NTT_VL(sew, lmul)>> {                         \
        ntt::vector<dtype, NTT_VL(sew, lmul)> operator()(                      \
            const ntt::vector<dtype, NTT_VL(sew, lmul)> &v1,                   \
            const ntt::vector<dtype, NTT_VL(sew, lmul)> &v2) const noexcept {  \
            ntt::vector<dtype, NTT_VL(sew, lmul)> vout;                        \
            auto p1 = reinterpret_cast<const dtype *>(v1.elements().data());   \
            auto p2 = reinterpret_cast<const dtype *>(v2.elements().data());   \
            auto pout = reinterpret_cast<dtype *>(vout.elements().data());     \
            auto input1 = vle##sew##_v_##dtype_prefix##sew##m##lmul(           \
                p1, NTT_VL(sew, lmul));                                        \
            auto input2 = vle##sew##_v_##dtype_prefix##sew##m##lmul(           \
                p2, NTT_VL(sew, lmul));                                        \
            auto output = kernel(input1, input2, NTT_VL(sew, lmul));           \
            vse##sew##_v_##dtype_prefix##sew##m##lmul(pout, output,            \
                                                      NTT_VL(sew, lmul));      \
                                                                               \
            return vout;                                                       \
        }                                                                      \
    };

// binary with float
#define REGISTER_RVV_BINARY_OP_FLOAT32(OP)                                     \
    RVV_BINARY_OP(OP, float, f, 32, 1, OP##_float32)                           \
    RVV_BINARY_OP(OP, float, f, 32, 2, OP##_float32)                           \
    RVV_BINARY_OP(OP, float, f, 32, 4, OP##_float32)                           \
    RVV_BINARY_OP(OP, float, f, 32, 8, OP##_float32)

// binary with int32_t
#define REGISTER_RVV_BINARY_OP_INT32(OP)                                       \
    RVV_BINARY_OP(OP, int32_t, i, 32, 1, OP##_int32)                           \
    RVV_BINARY_OP(OP, int32_t, i, 32, 2, OP##_int32)                           \
    RVV_BINARY_OP(OP, int32_t, i, 32, 4, OP##_int32)                           \
    RVV_BINARY_OP(OP, int32_t, i, 32, 8, OP##_int32)

// add
#define ADD_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t add_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfadd_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(ADD_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(add)

// sub
#define SUB_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t sub_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfsub_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(SUB_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(sub)

// mul
#define MUL_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t mul_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmul_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MUL_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(mul)

// div
#define DIV_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t div_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfdiv_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(DIV_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(div)

// mod
#define MOD_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t mod_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        auto quotient = vfcvt_f_x_v_f32m##LMUL(                                \
            vfcvt_rtz_x_f_v_i32m##LMUL(vfdiv_vv_f32m##LMUL(v1, v2, vl), vl),   \
            vl);                                                               \
        return vfsub_vv_f32m##LMUL(v1, vfmul_vv_f32m##LMUL(v2, quotient, vl),  \
                                   vl);                                        \
    }
IMPL_RVV_WITH_LMULS(MOD_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(mod)

// min
#define MIN_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t min_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmin_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MIN_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(min)

// max
#define MAX_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t max_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return vfmax_vv_f32m##LMUL(v1, v2, vl);                                \
    }

IMPL_RVV_WITH_LMULS(MAX_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(max)

// pow
#define POW_FLOAT32(LMUL, MLEN)                                                \
    inline vfloat32m##LMUL##_t pow_float32(const vfloat32m##LMUL##_t &v1,      \
                                           const vfloat32m##LMUL##_t &v2,      \
                                           const size_t vl) {                  \
        return pow_ps(v1, v2, vl);                                             \
    }

IMPL_RVV_WITH_LMULS(POW_FLOAT32)
REGISTER_RVV_BINARY_OP_FLOAT32(pow)

// floor_mod
#define FLOOR_MOD_INT32(LMUL, MLEN)                                            \
    inline vint32m##LMUL##_t floor_mod_int32(const vint32m##LMUL##_t &v1,      \
                                             const vint32m##LMUL##_t &v2,      \
                                             const size_t vl) {                \
        auto f1 = vfcvt_f_x_v_f32m##LMUL(v1, vl);                              \
        auto f2 = vfcvt_f_x_v_f32m##LMUL(v2, vl);                              \
                                                                               \
        auto vf = vfdiv_vv_f32m##LMUL(f1, f2, vl);                             \
        auto vi = vfcvt_x_f_v_i32m##LMUL(vf, vl);                              \
        auto _mask = vmfgt_vv_f32m##LMUL##_b##MLEN(                            \
            vfcvt_f_x_v_f32m##LMUL(vi, vl), vf, vl);                           \
        auto quotient = vsub_vx_i32m##LMUL##_m(_mask, vi, vi, 1, vl);          \
        return vsub_vv_i32m##LMUL(v1, vmul_vv_i32m##LMUL(quotient, v2, vl),    \
                                  vl);                                         \
    }

IMPL_RVV_WITH_LMULS(FLOOR_MOD_INT32)
REGISTER_RVV_BINARY_OP_INT32(floor_mod)

#endif
} // namespace nncase::ntt::ops
