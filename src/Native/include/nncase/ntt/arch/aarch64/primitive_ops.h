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
#include "arm_math.h"

namespace nncase::ntt::ops {

// unary op

// abs
// template <> struct abs<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return (v);
//     }
// };

// acos
// template <> struct acos<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return acos256_ps(v);
//     }
// };

// acosh(v) = ln(v + sqrt(v^2 - 1)), v >= 1
// template <> struct acosh<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         auto ones = _mm256_set1_ps(1.0f);
//         return log256_ps(_mm256_add_ps(
//             v, _mm256_sqrt_ps(_mm256_sub_ps(_mm256_mul_ps(v, v), ones))));
//     }
// };

// asin
// template <> struct asin<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return asin256_ps(v);
//     }
// };

// asinh(v) = ln(v + sqrt(v^2 + 1))
// template <> struct asinh<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         auto ones = _mm256_set1_ps(1.0f);
//         return log256_ps(_mm256_add_ps(
//             v, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(v, v), ones))));
//     }
// };

// ceil
// template <> struct ceil<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return _mm256_ceil_ps(v);
//     }
// };

// cos
template <> struct cos<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        return float32x4x2_t{cos_ps(v.val[0]), cos_ps(v.val[1])};
    }
};

// cosh(v) = (exp(v) + exp(-v)) / 2
// template <> struct cosh<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         auto zeros = _mm256_setzero_ps();
//         auto twos = _mm256_set1_ps(2.0f);
//         return _mm256_div_ps(
//             _mm256_add_ps(exp256_ps(v), exp256_ps(_mm256_sub_ps(zeros, v))),
//             twos);
//     }
// };

// exp
template <> struct exp<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        return float32x4x2_t{exp_ps(v.val[0]), exp_ps(v.val[1])};
    }
};

// floor
// template <> struct floor<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return _mm256_floor_ps(v);
//     }
// };

// log
// template <> struct log<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return log256_ps(v);
//     }
// };

// neg
template <> struct neg<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        auto zero = vdupq_n_f32(0);
        return float32x4x2_t{zero - v.val[0], zero - v.val[1]};
    }
};

// round
// template <> struct round<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return _mm256_round_ps(v,
//                                _MM_FROUND_TO_NEAREST_INT |
//                                _MM_FROUND_NO_EXC);
//     }
// };

// rsqrt
// template <> struct rsqrt<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return _mm256_rsqrt_ps(v);
//     }
// };

// sign
// template <> struct sign<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
// #if 0
//         auto sign_mask = _mm256_set1_ps(-0.0f);
//         auto sign_bits = _mm256_and_ps(v, sign_mask);
//         auto minus_ones = _mm256_set1_ps(-1.0f);
//         auto zeros = _mm256_setzero_ps();
//         auto ret = _mm256_blendv_ps(zeros, minus_ones, sign_bits);
//         auto gt_zero_mask = _mm256_cmp_ps(v, zeros, _CMP_GT_OQ);
//         auto ones = _mm256_set1_ps(1.0f);
//         ret = _mm256_blendv_ps(ret, ones, gt_zero_mask);
// #else
//         auto minus_ones = _mm256_set1_ps(-1.0f);
//         auto ones = _mm256_set1_ps(1.0f);
//         auto zeros = _mm256_setzero_ps();
//         auto ret = _mm256_setzero_ps();
//         auto mask = _mm256_cmp_ps(v, zeros, _CMP_GT_OQ);
//         ret = _mm256_blendv_ps(ret, ones, mask);
//         mask = _mm256_cmp_ps(v, zeros, _CMP_LT_OQ);
//         ret = _mm256_blendv_ps(ret, minus_ones, mask);
// #endif
//         return ret;
//     }
// };

// sin
// template <> struct sin<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return sin256_ps(v);
//     }
// };

// sinh(v) = (exp(v) - exp(-v)) / 2
// template <> struct sinh<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         auto zeros = _mm256_setzero_ps();
//         auto twos = _mm256_set1_ps(2.0f);
//         return _mm256_div_ps(
//             _mm256_sub_ps(exp256_ps(v), exp256_ps(_mm256_sub_ps(zeros, v))),
//             twos);
//     }
// };

// sqrt
template <> struct sqrt<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        return float32x4x2_t{vsqrtq_f32(v.val[0]), vsqrtq_f32(v.val[1])};
    }
};

// square
template <> struct square<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        return float32x4x2_t{v.val[0] * v.val[0], v.val[1] * v.val[1]};
    }
};

// tanh
// template <> struct tanh<ntt::vector<float, 8>> {
//     ntt::vector<float, 8> operator()(const ntt::vector<float, 8>& v) const
//     noexcept
//     {
//         return tanh256_ps(v);
//     }
// };

// swish(v) = v / (1 + std::exp(-v))
template <> struct swish<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(const ntt::vector<float, 8> &v) const noexcept {
        return impl(v);
    }

    inline float32x4x2_t impl(const float32x4x2_t &v) const {
        return float32x4x2_t{impl2(v.val[0]), impl2(v.val[1])};
    }

    inline float32x4_t impl2(const float32x4_t &v) const noexcept {
        auto zero = vdupq_n_f32(0);
        auto one = vdupq_n_f32(1);
        return v / exp_ps(zero - v) + one;
    }
};

template <> struct swish<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &v) const noexcept {
        return impl(v);
    }

    inline float32x4_t impl(const float32x4_t &v) const noexcept {
        auto zero = vdupq_n_f32(0);
        auto one = vdupq_n_f32(1);
        return v / (one + exp_ps(zero - v));
    }
};

// binary
template <> struct add<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vaddq_f32(lhs, rhs);
    }
};

template <> struct add<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(ntt::vector<float, 8> lhs,
               ntt::vector<float, 8> rhs) const noexcept {
        return impl(lhs, rhs);
    }
    inline float32x4x2_t impl(const float32x4x2_t &lhs,
                              const float32x4x2_t &rhs) const {
        return float32x4x2_t{vaddq_f32(lhs.val[0], rhs.val[0]),
                             vaddq_f32(lhs.val[1], rhs.val[1])};
    }
};

template <> struct sub<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vsubq_f32(lhs, rhs);
    }
};

template <> struct sub<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(ntt::vector<float, 8> lhs,
               ntt::vector<float, 8> rhs) const noexcept {
        return impl(lhs, rhs);
    }
    inline float32x4x2_t impl(const float32x4x2_t &lhs,
                              const float32x4x2_t &rhs) const {
        return float32x4x2_t{vsubq_f32(lhs.val[0], rhs.val[0]),
                             vsubq_f32(lhs.val[1], rhs.val[1])};
    }
};

template <> struct mul<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vmulq_f32(lhs, rhs);
    }
};

template <> struct mul<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(ntt::vector<float, 8> lhs,
               ntt::vector<float, 8> rhs) const noexcept {
        return impl(lhs, rhs);
    }

    inline float32x4x2_t impl(const float32x4x2_t &lhs,
                              const float32x4x2_t &rhs) const {
        return float32x4x2_t{vmulq_f32(lhs.val[0], rhs.val[0]),
                             vmulq_f32(lhs.val[1], rhs.val[1])};
    }
};

template <> struct div<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vdivq_f32(lhs, rhs);
    }
};

template <> struct div<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(ntt::vector<float, 8> lhs,
               ntt::vector<float, 8> rhs) const noexcept {
        return impl(lhs, rhs);
    }
    inline float32x4x2_t impl(const float32x4x2_t &lhs,
                              const float32x4x2_t &rhs) const {
        return float32x4x2_t{vdivq_f32(lhs.val[0], rhs.val[0]),
                             vdivq_f32(lhs.val[1], rhs.val[1])};
    }
};

template <> struct max<ntt::vector<float, 4>> {
    inline ntt::vector<float, 4>
    operator()(const ntt::vector<float, 4> &lhs,
               const ntt::vector<float, 4> &rhs) const noexcept {
        return vmaxq_f32(lhs, rhs);
    }
};

template <> struct max<ntt::vector<float, 8>> {
    inline ntt::vector<float, 8>
    operator()(ntt::vector<float, 8> lhs,
               ntt::vector<float, 8> rhs) const noexcept {
        return impl(lhs, rhs);
    }
    inline float32x4x2_t impl(const float32x4x2_t &lhs,
                              const float32x4x2_t &rhs) const {
        return float32x4x2_t{vmaxq_f32(lhs.val[0], rhs.val[0]),
                             vmaxq_f32(lhs.val[1], rhs.val[1])};
    }
};

} // namespace nncase::ntt::ops
