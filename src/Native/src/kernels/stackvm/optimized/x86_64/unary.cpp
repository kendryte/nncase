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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

#if defined(X86_64_SIMD_ON)
#include "avx_mathfun.h"
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
using namespace nncase::runtime::stackvm;

struct unary_op_abs {
    unary_op_abs() {
#if defined(X86_64_SIMD_ON)
        const ALIGN32_BEG int32_t remove_sign_bit_data[8] ALIGN32_END = {
            0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
            0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        remove_sign_bit_flag =
            _mm256_load_si256((__m256i const *)remove_sign_bit_data);
#endif
    }

    float operator()(float x) const { return fabsf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256i vector_a = _mm256_loadu_si256((__m256i const *)a);
        __m256i dst_a = _mm256_and_si256(vector_a, remove_sign_bit_flag);
        _mm256_storeu_si256((__m256i *)b, dst_a);
    }

  private:
    __m256i remove_sign_bit_flag;
#endif
};

struct unary_op_ceil {
    float operator()(float x) const { return ceilf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_cos {
    float operator()(float x) const { return cosf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = cos256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_exp {
    float operator()(float x) const { return expf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = exp256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_floor {
    float operator()(float x) const { return floorf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_log {
    float operator()(float x) const { return logf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = log256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_neg {
    float operator()(float x) const { return -(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sub_ps(_mm256_setzero_ps(), vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_round {
    float operator()(float x) const { return roundf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_rsqrt {
    float operator()(float x) const { return 1.0f / sqrtf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_rsqrt_ps(aa);
        _mm256_storeu_ps(b, bb);
    }
#endif
};

struct unary_op_sign {
    float operator()(float x) const { return (0.f < x) - (x < 0.f); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 b1 = _mm256_cmp_ps(_mm256_setzero_ps(), aa, _CMP_LT_OQ);
        __m256 b2 = _mm256_cmp_ps(aa, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256i ib1 = _mm256_castps_si256(b1);
        __m256i ib2 = _mm256_castps_si256(b2);
        __m256i ret = _mm256_sub_epi32(ib2, ib1);

        __m256 kbb = _mm256_cvtepi32_ps(ret);
        _mm256_storeu_ps(b, kbb);
    }
#endif
};

struct unary_op_sin {
    float operator()(float x) const { return sinf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = sin256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_sqrt {
    float operator()(float x) const { return sqrtf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sqrt_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_square {
    float operator()(float x) const { return x * x; }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_mul_ps(vector_a, vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

struct unary_op_tanh {
    float operator()(float x) const { return tanhf(x); }

#if defined(X86_64_SIMD_ON)
    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = tanh256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
#endif
};

template <typename Top>
result<void> optimized_unary_impl(const float *CXX_RESTRICT input,
                                  float *CXX_RESTRICT output,
                                  const dims_t &shape) noexcept {
    Top op;
    size_t n = compute_size(shape);
#if defined(X86_64_SIMD_ON)
    size_t n8 = (n >> 3);
    size_t n8_left = n & (8 - 1);
    for (size_t i = 0; i < n8; i++) {
        op.pack(input, output);
        input += 8;
        output += 8;
    }

    for (size_t i = 0; i < n8_left; i++) {
        output[i] = op(input[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        output[i] = op(input[i]);
    }
#endif

    return ok();
}

result<void> optimized::unary(typecode_t dtype, runtime::stackvm::unary_op_t op,
                              const gsl::byte *in, gsl::byte *out,
                              const dims_t &shape, const strides_t &in_strides,
                              const dims_t &out_shape,
                              const strides_t &out_strides,
                              kernel_context &context) noexcept {
    auto *input = IN_CAST(float, in);
    auto *output = OUT_CAST(float, out);

    switch (op) {
    case unary_op_t::abs: {
        return optimized_unary_impl<unary_op_abs>(input, output, shape);
    }
    case unary_op_t::ceil: {
        return optimized_unary_impl<unary_op_ceil>(input, output, shape);
    }
    case unary_op_t::cos: {
        return optimized_unary_impl<unary_op_cos>(input, output, shape);
    }
    case unary_op_t::exp: {
        return optimized_unary_impl<unary_op_exp>(input, output, shape);
    }
    case unary_op_t::floor: {
        return optimized_unary_impl<unary_op_floor>(input, output, shape);
    }
    case unary_op_t::log: {
        return optimized_unary_impl<unary_op_log>(input, output, shape);
    }
    case unary_op_t::neg: {
        return optimized_unary_impl<unary_op_neg>(input, output, shape);
    }
    case unary_op_t::round: {
        return optimized_unary_impl<unary_op_round>(input, output, shape);
    }
    case unary_op_t::rsqrt: {
        return optimized_unary_impl<unary_op_rsqrt>(input, output, shape);
    }
    case unary_op_t::sign: {
        return optimized_unary_impl<unary_op_sign>(input, output, shape);
    }
    case unary_op_t::sin: {
        return optimized_unary_impl<unary_op_sin>(input, output, shape);
    }
    case unary_op_t::sqrt: {
        return optimized_unary_impl<unary_op_sqrt>(input, output, shape);
    }
    case unary_op_t::square: {
        return optimized_unary_impl<unary_op_square>(input, output, shape);
    }
    case unary_op_t::tanh: {
        return optimized_unary_impl<unary_op_tanh>(input, output, shape);
    }
    default:
        return stackvm::reference::unary(dtype, op, in, out, shape, in_strides,
                                         out_shape, out_strides, context);
    }

    return ok();
}
