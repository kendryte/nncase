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
#include "arch_types.h"
#include "avx_mathfun.h"

namespace nncase::ntt::ops {

#ifdef __AVX2__

// unary op

// abs
template <> struct abs<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return abs256_ps(v);
    }
};

// acos
template <> struct acos<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
#if 0
        return acos256_ps(v);
#else
        // 定义常量
        const __m256 zero = _mm256_set1_ps(0.0f);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256 minus_one = _mm256_set1_ps(-1.0f);

        // 定义多项式系数
        const __m256 p0 = _mm256_set1_ps(0x1.55555ep-3);
        const __m256 p1 = _mm256_set1_ps(0x1.33261ap-4);
        const __m256 p2 = _mm256_set1_ps(0x1.70d7dcp-5);

        // 计算符号掩码和绝对值
        const __m256 abs_mask = _mm256_set1_ps(-0.0f); // 位掩码，用于计算绝对值
        const __m256 neg_mask = _mm256_cmp_ps(v, zero, _CMP_LT_OS); // v < 0.0
        __m256 x = _mm256_andnot_ps(abs_mask, v);                   // 绝对值

        // 初始化偏移量和乘法因子
        const __m256 off =
            _mm256_blendv_ps(zero, _mm256_set1_ps(0x1.921fb6p+1f), neg_mask);
        const __m256 mul1 =
            _mm256_blendv_ps(two, _mm256_set1_ps(-2.0f), neg_mask);
        const __m256 mul2 = _mm256_blendv_ps(minus_one, one, neg_mask);

        // 判断 x 是否小于 0.5
        const __m256 le_half_mask = _mm256_cmp_ps(x, half, _CMP_LE_OS);
        __m256 tmp = x;
        __m256 mul = _mm256_blendv_ps(mul1, mul2, le_half_mask);

        // 计算多项式 Q(x)
        tmp = _mm256_fnmadd_ps(tmp, half, half); // tmp = half - half * tmp
        const __m256 add =
            _mm256_blendv_ps(off, _mm256_set1_ps(0x1.921fb6p+0f), le_half_mask);
        __m256 z2 = _mm256_mul_ps(v, v); // z2 = tmp * v
        z2 = _mm256_blendv_ps(tmp, z2, le_half_mask);

        // 计算多项式近似
        __m256 y1 = _mm256_set1_ps(0x1.3af7d8p-5);
        __m256 y2 = _mm256_set1_ps(0x1.b059dp-6);
        tmp = _mm256_sqrt_ps(z2);                // tmp = sqrt(z2)
        const __m256 z4 = _mm256_mul_ps(z2, z2); // z4 = z2 * z2

        y1 = _mm256_fmadd_ps(y1, z4, p2); // y1 = y1 * z4 + p2
        y2 = _mm256_fmadd_ps(y2, z4, p1); // y2 = y2 * z4 + p1
        y1 = _mm256_fmadd_ps(y1, z4, p0); // y1 = y1 * z4 + p0

        const __m256 z = _mm256_blendv_ps(tmp, x, le_half_mask);
        y1 = _mm256_fmadd_ps(y2, z2, y1);  // y1 = y1 * y2 + z2
        mul = _mm256_mul_ps(mul, z);       // mul = mul * z
        y1 = _mm256_fmadd_ps(y1, z2, one); // y1 = y1 * z2 + one

        // 计算最终结果并返回
        return _mm256_fmadd_ps(y1, mul, add); // y1 * mul + add
#endif
    }
};

// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
template <> struct acosh<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return log256_ps(_mm256_add_ps(v, _mm256_sqrt_ps(_mm256_comp_fmsub_ps(
                                              v, v, _mm256_set1_ps(1.0f)))));
    }
};

// asin
template <> struct asin<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return asin256_ps(v);
    }
};

// asinh(v) = ln(v + sqrt(v^2 + 1))
template <> struct asinh<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return log256_ps(_mm256_add_ps(v, _mm256_sqrt_ps(_mm256_comp_fmadd_ps(
                                              v, v, _mm256_set1_ps(1.0f)))));
    }
};

// ceil
template <> struct ceil<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_ceil_ps(v);
    }
};

// cos
template <> struct cos<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return cos256_ps(v);
    }
};

// cosh(v) = (exp(v) + exp(-v)) / 2
template <> struct cosh<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        auto expV = exp256_ps(v);
        auto expNV = _mm256_div_ps(_mm256_set1_ps(1.f), expV);
        return _mm256_mul_ps(_mm256_add_ps(expV, expNV), _mm256_set1_ps(0.50f));
    }
};

// exp
template <> struct exp<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return exp256_ps(v);
    }
};

// floor
template <> struct floor<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_floor_ps(v);
    }
};

// log
template <> struct log<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return log256_ps(v);
    }
};

// neg
template <> struct neg<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_sub_ps(_mm256_setzero_ps(), v);
    }
};

// round
template <> struct round<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_round_ps(v,
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
};

// rsqrt
template <> struct rsqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {

#if 0
        return _mm256_rsqrt_ps(v);

#else
        // This is a higher precision version, ulp is about 4, tp=4.5
        const __m256 one_point_five = _mm256_set1_ps(1.5f);

        // Convert float to int representation and perform the initial magic
        // step
        __m256i ux = _mm256_castps_si256(v);
        ux = _mm256_srli_epi32(ux, 1);
        ux = _mm256_sub_epi32(_mm256_set1_epi32(0x5f375a86), ux);
        __m256 y = _mm256_castsi256_ps(ux);

        // First iteration
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 x = _mm256_mul_ps(v, _mm256_set1_ps(-0.5f));
        y2 = _mm256_fmadd_ps(y2, x, one_point_five);
        y = _mm256_mul_ps(y, y2);

        // Second iteration
        y2 = _mm256_mul_ps(y, y);
        y2 = _mm256_fmadd_ps(y2, x, one_point_five);
        y = _mm256_mul_ps(y, y2);

        // third iteration
        y2 = _mm256_mul_ps(y, y);
        y2 = _mm256_fmadd_ps(y2, x, one_point_five);
        y = _mm256_mul_ps(y, y2);

        return y;
#endif
    }
};

// sign
template <> struct sign<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
#if 0
        auto sign_mask = _mm256_set1_ps(-0.0f);
        auto sign_bits = _mm256_and_ps(v, sign_mask);
        auto minus_ones = _mm256_set1_ps(-1.0f);
        auto zeros = _mm256_setzero_ps();
        auto ret = _mm256_blendv_ps(zeros, minus_ones, sign_bits);
        auto gt_zero_mask = _mm256_cmp_ps(v, zeros, _CMP_GT_OQ);
        auto ones = _mm256_set1_ps(1.0f);
        ret = _mm256_blendv_ps(ret, ones, gt_zero_mask);
#else
        auto minus_ones = _mm256_set1_ps(-1.0f);
        auto ones = _mm256_set1_ps(1.0f);
        auto zeros = _mm256_setzero_ps();
        auto ret = _mm256_setzero_ps();
        auto mask = _mm256_cmp_ps(v, zeros, _CMP_GT_OQ);
        ret = _mm256_blendv_ps(ret, ones, mask);
        mask = _mm256_cmp_ps(v, zeros, _CMP_LT_OQ);
        ret = _mm256_blendv_ps(ret, minus_ones, mask);
#endif
        return ret;
    }
};

// sin
template <> struct sin<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return sin256_ps(v);
    }
};

// sinh(v) = (exp(v) - exp(-v)) / 2
template <> struct sinh<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        auto expV = exp256_ps(v);
        auto expNV = _mm256_rcp_ps(expV);
        return _mm256_mul_ps(_mm256_sub_ps(expV, expNV), _mm256_set1_ps(0.50f));
    }
};

// sqrt
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_rcp_ps(_mm256_rsqrt_ps(v));
    }
};

// square
template <> struct square<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_mul_ps(v, v);
    }
};

// swish(v) = v / (exp(-v) + 1)
template <> struct swish<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_mul_ps(
            v, _mm256_rcp_ps(_mm256_add_ps(
                   exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), v)),
                   _mm256_set1_ps(1.0f))));
    }
};

// tanh
template <> struct tanh<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return tanh256_ps(v);
    }
};

// binary

// add
template <> struct add<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_add_ps(v1, v2);
    }
};

template <> struct add<ntt::vector<float, 8, 8>, ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8>
    operator()(const ntt::vector<float, 8, 8> &v1,
               const ntt::vector<float, 8, 8> &v2) const noexcept {
        ntt::vector<float, 8, 8> v3;
        auto lhs_v = (vector<float, 8> *)(v1.elements().data());
        auto rhs_v = (vector<float, 8> *)(v2.elements().data());
        auto output_v = (vector<float, 8> *)(v3.elements().data());

        output_v[0] = _mm256_add_ps(lhs_v[0], rhs_v[0]);
        output_v[1] = _mm256_add_ps(lhs_v[1], rhs_v[1]);
        output_v[2] = _mm256_add_ps(lhs_v[2], rhs_v[2]);
        output_v[3] = _mm256_add_ps(lhs_v[3], rhs_v[3]);
        output_v[4] = _mm256_add_ps(lhs_v[4], rhs_v[4]);
        output_v[5] = _mm256_add_ps(lhs_v[5], rhs_v[5]);
        output_v[6] = _mm256_add_ps(lhs_v[6], rhs_v[6]);
        output_v[7] = _mm256_add_ps(lhs_v[7], rhs_v[7]);

        return v3;
    }
};

// sub
template <> struct sub<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_sub_ps(v1, v2);
    }
};

// swishb(v) = v / (exp(-v*beta) + 1)
template <> struct swishb<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v,
               const ntt::vector<float, 8> &b) const noexcept {
        return _mm256_mul_ps(
            v, _mm256_rcp_ps(
                   _mm256_add_ps(exp256_ps(_mm256_mul_ps(
                                     _mm256_sub_ps(_mm256_setzero_ps(), v), b)),
                                 _mm256_set1_ps(1.0f))));
    }
};

// mul
template <> struct mul<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_mul_ps(v1, v2);
    }
};

// mul
template <> struct mul<ntt::vector<float, 8>, float> {
    ntt::vector<float, 8> operator()(const ntt::vector<float, 8> &v1,
                                     const float &f2) const noexcept {
        auto v2 = _mm256_set1_ps(f2);
        return _mm256_mul_ps(v1, v2);
    }
};

// mul
template <> struct mul<float, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const float &f1,
               const ntt::vector<float, 8> &v2) const noexcept {
        auto v1 = _mm256_set1_ps(f1);
        return _mm256_mul_ps(v1, v2);
    }
};

template <>
struct mul_add<ntt::vector<float, 8>, ntt::vector<float, 8>,
               ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1, const ntt::vector<float, 8> &v2,
               const ntt::vector<float, 8> &v3) const noexcept {
        return _mm256_fmadd_ps(v1, v2, v3);
    }
};

template <>
struct mul_add<float, ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const float &f1, const ntt::vector<float, 8> &v2,
               const ntt::vector<float, 8> &v3) const noexcept {
        auto v1 = _mm256_set1_ps(f1);
        return _mm256_fmadd_ps(v1, v2, v3);
    }
};

template <>
struct mul_add<ntt::vector<float, 8>, float, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1, const float &f2,
               const ntt::vector<float, 8> &v3) const noexcept {
        auto v2 = _mm256_set1_ps(f2);
        return _mm256_fmadd_ps(v1, v2, v3);
    }
};

// div
template <> struct div<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_div_ps(v1, v2);
    }
};

// floor_mod
template <> struct floor_mod<ntt::vector<int32_t, 8>, ntt::vector<int32_t, 8>> {
    ntt::vector<int32_t, 8>
    operator()(ntt::vector<int32_t, 8> v1,
               ntt::vector<int32_t, 8> v2) const noexcept {

        auto f1 = _mm256_cvtepi32_ps(v1);
        auto f2 = _mm256_cvtepi32_ps(v2);
        auto quotient = _mm256_floor_ps(_mm256_div_ps(f1, f2));
        auto remainder = _mm256_comp_fnmadd_ps(quotient, f2, f1);
        return _mm256_cvtps_epi32(remainder);
    }
};

// mod
template <> struct mod<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        auto quotient = _mm256_round_ps(_mm256_div_ps(v1, v2),
                                        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        return _mm256_comp_fnmadd_ps(quotient, v2, v1);
    }
};

template <> struct mod<ntt::vector<int32_t, 8>, ntt::vector<int32_t, 8>> {
    ntt::vector<int32_t, 8>
    operator()(ntt::vector<int32_t, 8> v1,
               ntt::vector<int32_t, 8> v2) const noexcept {
        auto f1 = _mm256_cvtepi32_ps(v1);
        auto f2 = _mm256_cvtepi32_ps(v2);
        auto quotient = _mm256_round_ps(_mm256_div_ps(f1, f2),
                                        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        auto remainder = _mm256_comp_fnmadd_ps(quotient, f2, f1);
        return _mm256_cvtps_epi32(remainder);
    }
};

// min
template <> struct min<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_min_ps(v1, v2);
    }
};

// max
template <> struct max<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_max_ps(v1, v2);
    }
};

template <bool AccC>
struct mma<AccC, ntt::vector<float, 8, 8>, ntt::vector<float, 8, 8>,
           ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8>
    operator()(const ntt::vector<float, 8, 8> &v1,
               const ntt::vector<float, 8, 8> &v2,
               const ntt::vector<float, 8, 8> &v3) const noexcept {
        auto lhs_v = (float *)(v1.elements().data());
        auto rhs_v = (vector<float, 8> *)(v2.elements().data());
        auto output_v = (vector<float, 8> *)(v3.elements().data());

        for (size_t k = 0; k < 8; k++) {
            output_v[0] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[0 * 8 + k], rhs_v[k], output_v[0])
                    : ntt::mul(lhs_v[0 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[1] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[1 * 8 + k], rhs_v[k], output_v[1])
                    : ntt::mul(lhs_v[1 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[2] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[2 * 8 + k], rhs_v[k], output_v[2])
                    : ntt::mul(lhs_v[2 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[3] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[3 * 8 + k], rhs_v[k], output_v[3])
                    : ntt::mul(lhs_v[3 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[4] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[4 * 8 + k], rhs_v[k], output_v[4])
                    : ntt::mul(lhs_v[4 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[5] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[5 * 8 + k], rhs_v[k], output_v[5])
                    : ntt::mul(lhs_v[5 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[6] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[6 * 8 + k], rhs_v[k], output_v[6])
                    : ntt::mul(lhs_v[6 * 8 + k], rhs_v[k]);
        }

        for (size_t k = 0; k < 8; k++) {
            output_v[7] =
                (k != 0 || AccC)
                    ? ntt::mul_add(lhs_v[7 * 8 + k], rhs_v[k], output_v[7])
                    : ntt::mul(lhs_v[7 * 8 + k], rhs_v[k]);
        }

        return v3;
    }
};

// pow
template <> struct pow<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return pow256_ps(v1, v2);
    }
};

// inner product
template <> struct inner_product<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    float operator()(const ntt::vector<float, 8> &v1,
                     const ntt::vector<float, 8> &v2) const noexcept {
        auto vec = _mm256_mul_ps(v1, v2);
        // Extract the lower 128-bit part
        auto low = _mm256_extractf128_ps(vec, 0);
        // Extract the upper 128-bit part
        auto high = _mm256_extractf128_ps(vec, 1);
        // Add the low and high parts
        auto sum128 = _mm_add_ps(low, high);

        // Horizontal add: sum the pairs of elements
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        // Extract the final sum from the 128-bit result
        return _mm_cvtss_f32(sum128);
    }
};

// outer product
template <> struct outer_product<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    fixed_tensor<float, 8, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        alignas(32) fixed_tensor<float, 8, 8> result;
        __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

        tmp0 = _mm256_set1_ps(((float *)&v1)[0]);
        tmp0 = _mm256_mul_ps(tmp0, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[0 * 8]), tmp0);

        tmp1 = _mm256_set1_ps(((float *)&v1)[1]);
        tmp1 = _mm256_mul_ps(tmp1, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[1 * 8]), tmp1);

        tmp2 = _mm256_set1_ps(((float *)&v1)[2]);
        tmp2 = _mm256_mul_ps(tmp2, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[2 * 8]), tmp2);

        tmp3 = _mm256_set1_ps(((float *)&v1)[3]);
        tmp3 = _mm256_mul_ps(tmp3, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[3 * 8]), tmp3);

        tmp4 = _mm256_set1_ps(((float *)&v1)[4]);
        tmp4 = _mm256_mul_ps(tmp4, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[4 * 8]), tmp4);

        tmp5 = _mm256_set1_ps(((float *)&v1)[5]);
        tmp5 = _mm256_mul_ps(tmp5, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[5 * 8]), tmp5);

        tmp6 = _mm256_set1_ps(((float *)&v1)[6]);
        tmp6 = _mm256_mul_ps(tmp6, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[6 * 8]), tmp6);

        tmp7 = _mm256_set1_ps(((float *)&v1)[7]);
        tmp7 = _mm256_mul_ps(tmp7, v2);
        _mm256_storeu_ps(&(((float *)(result.elements().data()))[7 * 8]), tmp7);

        return result;
    }
};

#endif
} // namespace nncase::ntt::ops
