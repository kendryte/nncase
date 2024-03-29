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
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return abs256_ps(v);
    }
};

// acos
template <> struct acos<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return acos256_ps(v);
    }
};

// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
template <> struct acosh<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        auto ones = _mm256_set1_ps(1.0f);
        return log256_ps(_mm256_add_ps(
            v, _mm256_sqrt_ps(_mm256_sub_ps(_mm256_mul_ps(v, v), ones))));
    }
};

// asin
template <> struct asin<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return asin256_ps(v);
    }
};

// asinh(v) = ln(v + sqrt(v^2 + 1))
template <> struct asinh<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        auto ones = _mm256_set1_ps(1.0f);
        return log256_ps(_mm256_add_ps(
            v, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(v, v), ones))));
    }
};

// ceil
template <> struct ceil<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_ceil_ps(v);
    }
};

// cos
template <> struct cos<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return cos256_ps(v);
    }
};

// cosh(v) = (exp(v) + exp(-v)) / 2
template <> struct cosh<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        auto zeros = _mm256_setzero_ps();
        auto twos = _mm256_set1_ps(2.0f);
        return _mm256_div_ps(
            _mm256_add_ps(exp256_ps(v), exp256_ps(_mm256_sub_ps(zeros, v))),
            twos);
    }
};

// exp
template <> struct exp<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return exp256_ps(v);
    }
};

// floor
template <> struct floor<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_floor_ps(v);
    }
};

// log
template <> struct log<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return log256_ps(v);
    }
};

// neg
template <> struct neg<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_sub_ps(_mm256_setzero_ps(), v);
    }
};

// round
template <> struct round<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_round_ps(v,
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
};

// rsqrt
template <> struct rsqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_rsqrt_ps(v);
    }
};

// sign
template <> struct sign<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
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
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return sin256_ps(v);
    }
};

// sinh(v) = (exp(v) - exp(-v)) / 2
template <> struct sinh<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        auto zeros = _mm256_setzero_ps();
        auto twos = _mm256_set1_ps(2.0f);
        return _mm256_div_ps(
            _mm256_sub_ps(exp256_ps(v), exp256_ps(_mm256_sub_ps(zeros, v))),
            twos);
    }
};

// sqrt
template <> struct sqrt<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_sqrt_ps(v);
    }
};

// square
template <> struct square<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return _mm256_mul_ps(v, v);
    }
};

// tanh
template <> struct tanh<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        return tanh256_ps(v);
    }
};

// swish(v) = v / (1 + std::exp(-v))
template <> struct swish<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v) const noexcept {
        auto ones = _mm256_set1_ps(1.0f);
        auto zeros = _mm256_setzero_ps();
        return _mm256_div_ps(
            v, _mm256_add_ps(ones, exp256_ps(_mm256_sub_ps(zeros, v))));
    }
};

#endif
} // namespace nncase::ntt::ops
