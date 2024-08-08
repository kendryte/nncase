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
        return acos256_ps(v);
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
        auto expNV = _mm256_rcp_ps(expV);
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
        return _mm256_rsqrt_ps(v);
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
    operator()(const ntt::vector<float, 8, 8> &lhs,
               const ntt::vector<float, 8, 8> &rhs,
               const ntt::vector<float, 8, 8> &v3) const noexcept {
        auto output = v3;
        for (size_t k = 0; k < 8; k++) {
            output(0) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(0, k), rhs(k), output(0))
                            : ntt::mul(lhs(0, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(1) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(1, k), rhs(k), output(1))
                            : ntt::mul(lhs(1, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(2) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(2, k), rhs(k), output(2))
                            : ntt::mul(lhs(2, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(3) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(3, k), rhs(k), output(3))
                            : ntt::mul(lhs(3, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(4) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(4, k), rhs(k), output(4))
                            : ntt::mul(lhs(4, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(5) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(5, k), rhs(k), output(5))
                            : ntt::mul(lhs(5, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(6) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(6, k), rhs(k), output(6))
                            : ntt::mul(lhs(6, k), rhs(k));
        }

        for (size_t k = 0; k < 8; k++) {
            output(7) = (k != 0 || AccC)
                            ? ntt::mul_add(lhs(7, k), rhs(k), output(7))
                            : ntt::mul(lhs(7, k), rhs(k));
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
    auto operator()(const ntt::vector<float, 8> &v1,
                    const ntt::vector<float, 8> &v2) const noexcept {
        ntt::vector<float, 8, 8> result;
        __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

        tmp0 = _mm256_set1_ps(v1(0));
        result(0) = _mm256_mul_ps(tmp0, v2);

        tmp1 = _mm256_set1_ps(v1(1));
        result(1) = _mm256_mul_ps(tmp1, v2);

        tmp2 = _mm256_set1_ps(v1(2));
        result(2) = _mm256_mul_ps(tmp2, v2);

        tmp3 = _mm256_set1_ps(v1(3));
        result(3) = _mm256_mul_ps(tmp3, v2);

        tmp4 = _mm256_set1_ps(v1(4));
        result(4) = _mm256_mul_ps(tmp4, v2);

        tmp5 = _mm256_set1_ps(v1(5));
        result(5) = _mm256_mul_ps(tmp5, v2);

        tmp6 = _mm256_set1_ps(v1(6));
        result(6) = _mm256_mul_ps(tmp6, v2);

        tmp7 = _mm256_set1_ps(v1(7));
        result(7) = _mm256_mul_ps(tmp7, v2);
        return result;
    }
};

#endif
} // namespace nncase::ntt::ops
