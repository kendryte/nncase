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
#include "avx_mathfun.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
using namespace nncase::runtime::stackvm;

struct unary_op_abs {
    unary_op_abs() : sign_bit_(_mm256_set1_ps(-0.0f)) {}

    float operator()(float x) const { return fabsf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_andnot_ps(sign_bit_, vector_a);
        _mm256_storeu_ps(b, dst_a);
    }

  private:
    __m256 sign_bit_;
};

struct unary_op_ceil {
    float operator()(float x) const { return ceilf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_cos {
    float operator()(float x) const { return cosf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = cos256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_exp {
    float operator()(float x) const { return expf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = exp256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_floor {
    float operator()(float x) const { return floorf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_log {
    float operator()(float x) const { return logf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = log256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_neg {
    float operator()(float x) const { return -(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sub_ps(_mm256_setzero_ps(), vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

static float round_onnx(float v) {
    if (v > 0 && v - (int32_t)v == 0.5) {
        float result = (int32_t)v + 1.0;
        if ((int32_t)result % 2 == 0)
            return result;
        else
            return result - 1;
    } else if (v < 0 && (int32_t)v - v == 0.5) {
        float result = (int32_t)v + 1.0;
        if ((int32_t)result % 2 == 0)
            return result;
        else
            return result - 1;
    } else
        return roundf(v);
}

struct unary_op_round {
    float operator()(float x) const { return round_onnx(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(
            vector_a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_rsqrt {
    float operator()(float x) const { return 1.0f / sqrtf(x); }

    void pack(const float *a, float *b) {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_rsqrt_ps(aa);
        _mm256_storeu_ps(b, bb);
    }
};

struct unary_op_sign {
    unary_op_sign() : zero_(_mm256_setzero_ps()) {}

    float operator()(float x) const { return (0.f < x) - (x < 0.f); }

    void pack(const float *a, float *b) {
        __m256 va = _mm256_loadu_ps(a);
        __m256 positive = _mm256_and_ps(_mm256_cmp_ps(zero_, va, _CMP_LT_OQ),
                                        _mm256_set1_ps(1.0f));
        __m256 negative = _mm256_and_ps(_mm256_cmp_ps(va, zero_, _CMP_LT_OQ),
                                        _mm256_set1_ps(-1.0f));
        __m256 vb = _mm256_or_ps(positive, negative);
        _mm256_storeu_ps(b, vb);
    }

  private:
    __m256 zero_;
};

struct unary_op_sin {
    float operator()(float x) const { return sinf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = sin256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_sqrt {
    float operator()(float x) const { return sqrtf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sqrt_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_square {
    float operator()(float x) const { return x * x; }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_mul_ps(vector_a, vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

struct unary_op_tanh {
    float operator()(float x) const { return tanhf(x); }

    void pack(const float *a, float *b) {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = tanh256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
    }
};

template <typename Top>
result<void> optimized_unary_impl(const float *CXX_RESTRICT input,
                                  float *CXX_RESTRICT output,
                                  std::span<const size_t> shape) noexcept {
    Top op;
    size_t n = compute_size(shape);
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

    return ok();
}

result<void> optimized::unary(typecode_t dtype, runtime::stackvm::unary_op_t op,
                              const std::byte *in, std::byte *out,
                              std::span<const size_t> shape,
                              std::span<const size_t> in_strides,
                              std::span<const size_t> out_shape,
                              std::span<const size_t> out_strides,
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
