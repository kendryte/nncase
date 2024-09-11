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
#if 0
        return asin256_ps(v);
#else
        // 定义常量
        const __m256 zero = _mm256_set1_ps(0.0f);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 minus_two = _mm256_set1_ps(-2.0f);
        const __m256 pi_over_2f = _mm256_set1_ps(0x1.921fb6p+0f);

        // 定义多项式系数
        const __m256 p0 = _mm256_set1_ps(0x1.55555ep-3);
        const __m256 p1 = _mm256_set1_ps(0x1.33261ap-4);
        const __m256 p2 = _mm256_set1_ps(0x1.70d7dcp-5);

        // 计算符号掩码和绝对值
        const __m256 neg_mask = _mm256_cmp_ps(v, zero, _CMP_LT_OS); // v < 0.0
        const __m256 abs_mask = _mm256_set1_ps(-0.0f); // 位掩码，用于计算绝对值
        __m256 x = _mm256_andnot_ps(abs_mask, v); // 绝对值

        // 初始化偏移量和乘法因子
        const __m256 mul1 =
            _mm256_blendv_ps(one, _mm256_set1_ps(-1.0f), neg_mask);

        // 判断 x 是否小于 0.5
        const __m256 lt_half_mask = _mm256_cmp_ps(x, half, _CMP_LT_OS);
        __m256 tmp = x;
        __m256 mul2 = _mm256_blendv_ps(minus_two, one, lt_half_mask);

        // 计算多项式 Q(x)
        tmp = _mm256_fnmadd_ps(tmp, half, half); // tmp = half - half * tmp
        const __m256 add = _mm256_blendv_ps(pi_over_2f, zero, lt_half_mask);
        __m256 z2 = _mm256_mul_ps(v, v); // z2 = tmp * v
        z2 = _mm256_blendv_ps(tmp, z2, lt_half_mask);

        // 计算多项式近似
        __m256 y1 = _mm256_set1_ps(0x1.3af7d8p-5);
        __m256 y2 = _mm256_set1_ps(0x1.b059dp-6);
        const __m256 z4 = _mm256_mul_ps(z2, z2); // z4 = z2 * z2
        tmp = _mm256_sqrt_ps(z2);                // tmp = sqrt(z2)

        y1 = _mm256_fmadd_ps(y1, z4, p2); // y1 = y1 * z4 + p2
        y2 = _mm256_fmadd_ps(y2, z4, p1); // y2 = y2 * z4 + p1
        y1 = _mm256_fmadd_ps(y1, z4, p0); // y1 = y1 * z4 + p0

        const __m256 z = _mm256_blendv_ps(tmp, x, lt_half_mask);
        y1 = _mm256_fmadd_ps(y2, z2, y1); // y1 = y1 * y2 + z2
        z2 = _mm256_mul_ps(z2, z);        // mul = mul * z
        y1 = _mm256_fmadd_ps(y1, z2, z);  // y1 = y1 * z2 + one
        // 计算最终结果并返回
        y1 = _mm256_fmadd_ps(y1, mul2, add); // y1 * mul + add
        return _mm256_mul_ps(y1, mul1);      // mul = mul * z
#endif
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
#if 0
        return cos256_ps(v);
#else
        auto n = _mm256_set1_ps(0x1.45f306p-2f);
        auto half = _mm256_set1_ps(0.5f);
        auto c0 = _mm256_set1_ps(-0x1.555548p-3f);
        auto c2 = _mm256_set1_ps(-0x1.9f42eap-13f);

        // n = rint((|x|+pi/2)/pi) - 0.5
        auto r = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(
                                      0x7FFFFFFF))); // vfabs_v_f32m
        n = _mm256_fmadd_ps(n, r, half);             // vfmadd_vv_f32m
        auto ni = _mm256_cvtps_epi32(n);             // vfcvt_x_f_v_i32m
        n = _mm256_cvtepi32_ps(ni);                  // vfcvt_f_x_v_f32m
        auto odd =
            _mm256_add_epi32(ni, _mm256_set1_epi32(0x1.8p+23)); // vadd_vx_i32m
        n = _mm256_sub_ps(n, _mm256_set1_ps(0.5f));             // vfsub_vf_f32m
        odd = _mm256_slli_epi32(odd, 31);                       // vsll_vx_i32

        // r = |x| - n*pi  (range reduction into -pi/2 .. pi/2)
        r = _mm256_fnmadd_ps(_mm256_set1_ps(0x1.921fb6p+1f), n,
                             r); // vfnmsac_vf_f32m
        r = _mm256_fnmadd_ps(_mm256_set1_ps(-0x1.777a5cp-24f), n,
                             r); // vfnmsac_vf_f32m
        r = _mm256_fnmadd_ps(_mm256_set1_ps(-0x1.ee59dap-49f), n,
                             r); // vfnmsac_vf_f32m

        // y = sin(r)
        auto r2 = _mm256_mul_ps(r, r); // vfmul_vv_f32m
        auto y1 = _mm256_set1_ps(0x1.5b2e76p-19f);
        auto y2 = _mm256_set1_ps(0x1.110df4p-7f);
        y1 = _mm256_fmadd_ps(y1, r2, c2);   // vfmadd_vv_f32m
        y2 = _mm256_fmadd_ps(y2, r2, c0);   // vfmadd_vv_f32m
        auto r4 = _mm256_mul_ps(r2, r2);    // vfmul_vv_f32m
        auto r3 = _mm256_mul_ps(r2, r);     // vfmul_vv_f32m
        y1 = _mm256_fmadd_ps(y1, r4, y2);   // vfmadd_vv_f32m
        y1 = _mm256_fmadd_ps(y1, r3, r);    // vfmadd_vv_f32m
        auto tmp = _mm256_castps_si256(y1); // vreinterpret_v_f32m_i32m
        tmp = _mm256_xor_si256(tmp, odd);   // vxor_vv_i32m
        return _mm256_castsi256_ps(tmp);    // vreinterpret_v_i32m_f32m
#endif
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

template <> struct reduce<add, float, ntt::vector<float, 8>> {
    float operator()(const ntt::vector<float, 8> &v) const noexcept {
        // Sum the elements in the 256-bit vector directly
        __m128 sum =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));

        // Extract and return the final sum
        return _mm_cvtss_f32(sum);
    }
};

template <> struct reduce<max, float, ntt::vector<float, 8>> {
    float operator()(const ntt::vector<float, 8> &v) const noexcept {
        // Sum the elements in the 256-bit vector directly
        __m128 sum =
            _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        sum = _mm_max_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_max_ss(sum, _mm_shuffle_ps(sum, sum, 1));

        // Extract and return the final sum
        return _mm_cvtss_f32(sum);
    }
};

template <> struct reduce<min, float, ntt::vector<float, 8>> {
    float operator()(const ntt::vector<float, 8> &v) const noexcept {
        // Sum the elements in the 256-bit vector directly
        __m128 sum =
            _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        sum = _mm_min_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_min_ss(sum, _mm_shuffle_ps(sum, sum, 1));

        // Extract and return the final sum
        return _mm_cvtss_f32(sum);
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
#if 0
        return sin256_ps(v);
#else
        // Define constants
        __m256 c0 = _mm256_set1_ps(-0x1.555548p-3f);
        __m256 c2 = _mm256_set1_ps(-0x1.9f42eap-13f);
        __m256 c3 = _mm256_set1_ps(0x1.45f306p-2f);
        __m256 c4 = _mm256_set1_ps(0x1.8p+23);
        __m256 c5 = _mm256_set1_ps(0x1.921fb6p+1f);
        __m256 c6 = _mm256_set1_ps(-0x1.777a5cp-24f);
        __m256 c7 = _mm256_set1_ps(-0x1.ee59dap-49f);
        __m256 c8 = _mm256_set1_ps(0x1.5b2e76p-19f);
        __m256 c9 = _mm256_set1_ps(0x1.110df4p-7f);

        // n = rint(|x|/pi)
        __m256 r = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); // fabs(v)
        __m256 n = _mm256_mul_ps(r, c3);
        __m256 sign = _mm256_and_ps(v, _mm256_set1_ps(-0.0f)); // sign bit
        __m256i ni = _mm256_cvtps_epi32(n);
        n = _mm256_cvtepi32_ps(ni);
        __m256i odd = _mm256_add_epi32(ni, _mm256_castps_si256(c4));

        // r = |x| - n*pi  (range reduction into -pi/2 .. pi/2)
        r = _mm256_fnmadd_ps(c5, n, r);
        odd = _mm256_slli_epi32(odd, 31);
        r = _mm256_fnmadd_ps(c6, n, r);
        r = _mm256_fnmadd_ps(c7, n, r);

        // y = sin(r)
        __m256 r2 = _mm256_mul_ps(r, r);
        __m256 y1 = c8;
        __m256 y2 = c9;
        y1 = _mm256_fmadd_ps(y1, r2, c2);
        y2 = _mm256_fmadd_ps(y2, r2, c0);
        __m256 r4 = _mm256_mul_ps(r2, r2);
        __m256 r3 = _mm256_mul_ps(r2, r);
        y1 = _mm256_fmadd_ps(y1, r4, y2);
        __m256 sign_adjust = _mm256_castsi256_ps(
            _mm256_xor_si256(_mm256_castps_si256(sign), odd));
        y1 = _mm256_fmadd_ps(y1, r3, r);
        __m256 tmp = _mm256_xor_ps(y1, sign_adjust);
        return tmp;
#endif
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
#if 0
// max_ulp_error = 8861
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_rcp_ps(_mm256_rsqrt_ps(v));
    }
};
#else
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return _mm256_sqrt_ps(v);
    }
};
#endif

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

// min(v1, v2)
template <> struct min<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v1,
               const ntt::vector<float, 8> &v2) const noexcept {
        return _mm256_min_ps(v1, v2);
    }
};

// min(v1, f2)
template <> struct min<ntt::vector<float, 8>, float> {
    ntt::vector<float, 8> operator()(const ntt::vector<float, 8> &v1,
                                     const float &f2) const noexcept {
        auto v2 = _mm256_set1_ps(f2);
        return _mm256_min_ps(v1, v2);
    }
};

// min(f1, v2)
template <> struct min<float, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const float &f1,
               const ntt::vector<float, 8> &v2) const noexcept {
        auto v1 = _mm256_set1_ps(f1);
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

// max(v1, f2)
template <> struct max<ntt::vector<float, 8>, float> {
    ntt::vector<float, 8> operator()(const ntt::vector<float, 8> &v1,
                                     const float &f2) const noexcept {
        auto v2 = _mm256_set1_ps(f2);
        return _mm256_max_ps(v1, v2);
    }
};

// max(f1, v2)
template <> struct max<float, ntt::vector<float, 8>> {
    ntt::vector<float, 8>
    operator()(const float &f1,
               const ntt::vector<float, 8> &v2) const noexcept {
        auto v1 = _mm256_set1_ps(f1);
        return _mm256_max_ps(v1, v2);
    }
};

template <bool AccC>
struct mma<AccC, ntt::vector<float, 8, 8>, ntt::vector<float, 8, 8>,
           ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8>
    operator()(const ntt::vector<float, 8, 8> &lhs,
               const ntt::vector<float, 8, 8> &rhs,
               const ntt::vector<float, 8, 8> &v3) const noexcept {
        ntt::vector<float, 8, 8> output;
        for (size_t k = 0; k < 8; k++) {
            for (size_t m = 0; m < 8; m++) {
                output(m) = (k != 0 || AccC)
                                ? ntt::mul_add(lhs(m, k), rhs(k),
                                               k == 0 ? v3(m) : output(m))
                                : ntt::mul(lhs(m, k), rhs(k));
            }
        }
        return output;
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
        // Multiply the elements
        __m256 mul = _mm256_mul_ps(v1, v2);

        // Sum the elements in the 256-bit vector directly
        __m128 sum1 = _mm_add_ps(_mm256_castps256_ps128(mul),
                                 _mm256_extractf128_ps(mul, 1));
        sum1 = _mm_add_ps(sum1, _mm_movehl_ps(sum1, sum1));
        sum1 = _mm_add_ss(sum1, _mm_shuffle_ps(sum1, sum1, 1));

        // Extract and return the final sum
        return _mm_cvtss_f32(sum1);
    }
};

// outer product
template <> struct outer_product<ntt::vector<float, 8>, ntt::vector<float, 8>> {
    auto operator()(const ntt::vector<float, 8> &v1,
                    const ntt::vector<float, 8> &v2) const noexcept {
        ntt::vector<float, 8, 8> result;
        for (size_t i = 0; i < 8; i++) {
            auto a_broadcast = _mm256_set1_ps(v1(i));
            result(i) = _mm256_mul_ps(a_broadcast, v2);
        }
        return result;
    }
};

template <> struct clamp<ntt::vector<float, 8>, float> {
    auto operator()(const ntt::vector<float, 8> &v, const float &min,
                    const float &max) const noexcept {
        auto tmp = _mm256_max_ps(v, _mm256_set1_ps(min));
        return _mm256_min_ps(tmp, _mm256_set1_ps(max));
    }
};

#endif
} // namespace nncase::ntt::ops
