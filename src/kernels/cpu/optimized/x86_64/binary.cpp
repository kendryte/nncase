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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

#define binary_operator_vec(op, type, typename)                                                                                                   \
    static void binary_##op##_##type##_vec(const typename *a, int len_a, const typename *b, int len_b, typename *c, int len_c, int transposition) \
    {                                                                                                                                             \
        (void)len_c;                                                                                                                              \
        (void)transposition;                                                                                                                      \
        if (len_a == len_b)                                                                                                                       \
        {                                                                                                                                         \
            op##_##type##_vv(a, b, c, len_a);                                                                                                     \
        }                                                                                                                                         \
        else if (len_a == 1)                                                                                                                      \
        {                                                                                                                                         \
            op##_##type##_vf(b, a[0], c, len_b);                                                                                                  \
        }                                                                                                                                         \
        else                                                                                                                                      \
        {                                                                                                                                         \
            op##_##type##_vf(a, b[0], c, len_a);                                                                                                  \
        }                                                                                                                                         \
    }

#define binary_operator_transposition_vec(op, type, typename)                                                                                     \
    static void binary_##op##_##type##_vec(const typename *a, int len_a, const typename *b, int len_b, typename *c, int len_c, int transposition) \
    {                                                                                                                                             \
        (void)len_c;                                                                                                                              \
        (void)transposition;                                                                                                                      \
        if (transposition)                                                                                                                        \
        {                                                                                                                                         \
            if (len_a == len_b)                                                                                                                   \
            {                                                                                                                                     \
                op##_##type##_vv(b, a, c, len_a);                                                                                                 \
            }                                                                                                                                     \
            else if (len_a == 1)                                                                                                                  \
            {                                                                                                                                     \
                op##_##type##_vf(b, a[0], c, len_b);                                                                                              \
            }                                                                                                                                     \
            else                                                                                                                                  \
            {                                                                                                                                     \
                op##_##type##_fv(b[0], a, c, len_a);                                                                                              \
            }                                                                                                                                     \
        }                                                                                                                                         \
        else                                                                                                                                      \
        {                                                                                                                                         \
            if (len_a == len_b)                                                                                                                   \
            {                                                                                                                                     \
                op##_##type##_vv(a, b, c, len_a);                                                                                                 \
            }                                                                                                                                     \
            else if (len_a == 1)                                                                                                                  \
            {                                                                                                                                     \
                op##_##type##_fv(a[0], b, c, len_b);                                                                                              \
            }                                                                                                                                     \
            else                                                                                                                                  \
            {                                                                                                                                     \
                op##_##type##_vf(a, b[0], c, len_a);                                                                                              \
            }                                                                                                                                     \
        }                                                                                                                                         \
    }

#define add_fun(a, b) ((a) + (b))
#define sub_fun(a, b) ((a) - (b))
#define mul_fun(a, b) ((a) * (b))
#define div_fun(a, b) ((a) / (b))
#define max_fun(a, b) ((a) > (b) ? (a) : (b))
#define min_fun(a, b) ((a) < (b) ? (a) : (b))
#define pow_fun(a, b) (pow((a), (b)))
#define logical_and_fun(a, b) ((a) && (b))

#define operator_vec(op, type, typename, fun_op)                                           \
    static void op##_##type##_vv(const typename *a, const typename *b, typename *c, int n) \
    {                                                                                      \
        for (int j = 0; j < n; ++j)                                                        \
        {                                                                                  \
            c[j] = fun_op(a[j], b[j]);                                                     \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    static void op##_##type##_vf(const typename *a, const typename b, typename *c, int n)  \
    {                                                                                      \
        for (int j = 0; j < n; ++j)                                                        \
        {                                                                                  \
            c[j] = fun_op(a[j], b);                                                        \
        }                                                                                  \
    }                                                                                      \
                                                                                           \
    static void op##_##type##_fv(const typename a, const typename *b, typename *c, int n)  \
    {                                                                                      \
        for (int j = 0; j < n; ++j)                                                        \
        {                                                                                  \
            c[j] = fun_op(a, b[j]);                                                        \
        }                                                                                  \
    }

#if !defined(X86_64_SIMD_ON)
#include "avx_mathfun.h"

static void add_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_add_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void add_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_add_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] + b;
    }
}

static void add_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i aa = _mm256_loadu_si256((__m256i const *)a);
        __m256i bb = _mm256_loadu_si256((__m256i const *)b);
        __m256i cc = _mm256_add_epi32(aa, bb);
        _mm256_storeu_si256((__m256i *)c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void add_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    int n8 = (n >> 1);
    int n8_left = n & (2 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m128i aa = _mm_loadu_si128((__m128i const *)a);
        __m128i bb = _mm_loadu_si128((__m128i const *)b);
        __m128i cc = _mm_add_epi64(aa, bb);
        _mm_storeu_si128((__m128i *)c, cc);
        c += 2;
        a += 2;
        b += 2;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void sub_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_sub_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] - b[j];
    }
}

static void sub_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_sub_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] - b;
    }
}

static void sub_f32_fv(const float a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 aa = _mm256_set1_ps(a);
    for (int j = 0; j < n8; ++j)
    {
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_sub_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a - b[j];
    }
}

static void sub_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i aa = _mm256_loadu_si256((__m256i const *)a);
        __m256i bb = _mm256_loadu_si256((__m256i const *)b);
        __m256i cc = _mm256_sub_epi32(aa, bb);
        _mm256_storeu_si256((__m256i *)c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void sub_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    int n8 = (n >> 1);
    int n8_left = n & (2 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m128i aa = _mm_loadu_si128((__m128i const *)a);
        __m128i bb = _mm_loadu_si128((__m128i const *)b);
        __m128i cc = _mm_sub_epi64(aa, bb);
        _mm_storeu_si128((__m128i *)c, cc);
        c += 2;
        a += 2;
        b += 2;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] - b[j];
    }
}

static void mul_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_mul_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] * b[j];
    }
}

static void mul_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_mul_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] * b;
    }
}

static void mul_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i aa = _mm256_loadu_si256((__m256i const *)a);
        __m256i bb = _mm256_loadu_si256((__m256i const *)b);
        __m256i cc = _mm256_mullo_epi32(aa, bb);
        _mm256_storeu_si256((__m256i *)c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] * b[j];
    }
}

static void div_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_div_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] / b[j];
    }
}

static void div_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_div_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] / b;
    }
}

static void div_f32_fv(const float a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 aa = _mm256_set1_ps(a);
    for (int j = 0; j < n8; ++j)
    {
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_div_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a / b[j];
    }
}

static void div_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i ia = _mm256_loadu_si256((__m256i const *)a);
        __m256i ib = _mm256_loadu_si256((__m256i const *)b);
        __m256 fa = _mm256_cvtepi32_ps(ia);
        __m256 fb = _mm256_cvtepi32_ps(ib);
        __m256 fc = _mm256_div_ps(fa, fb);
        __m256i ic = _mm256_cvtps_epi32(fc);
        _mm256_storeu_si256((__m256i *)c, ic);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] / b[j];
    }
}

static void min_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_min_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] < b[j] ? a[j] : b[j];
    }
}

static void min_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_min_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] < b ? a[j] : b;
    }
}

static void max_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = _mm256_max_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] > b[j] ? a[j] : b[j];
    }
}

static void max_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = _mm256_max_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] > b ? a[j] : b;
    }
}

static void max_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i aa = _mm256_loadu_si256((__m256i const *)a);
        __m256i bb = _mm256_loadu_si256((__m256i const *)b);
        __m256i cc = _mm256_min_epi32(aa, bb);
        _mm256_storeu_si256((__m256i *)c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] > b[j] ? a[j] : b[j];
    }
}

static void pow_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = pow256_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] > b[j] ? a[j] : b[j];
    }
}

static void pow_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256 bb = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 cc = pow256_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        a += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a[j] > b ? a[j] : b;
    }
}

static void pow_f32_fv(const float a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 aa = _mm256_set1_ps(a);
    for (int j = 0; j < n8; ++j)
    {
        __m256 bb = _mm256_loadu_ps(b);
        __m256 cc = pow256_ps(aa, bb);
        _mm256_storeu_ps(c, cc);
        c += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = a / b[j];
    }
}

static CAN_FORCEINLINE int _mm256i_reduce_max_i32(__m256i x)
{
    const __m128i x128 = _mm_max_epi32(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(x, 1));
    const __m128i x64 = _mm_max_epi32(x128, _mm_cvtps_epi32(_mm_movehl_ps(_mm_cvtepi32_ps(x128), _mm_cvtepi32_ps(x128))));
    const __m128i x32 = _mm_max_epi32(x64, _mm_shuffle_epi32(x64, 0x55));
    return _mm_cvtsi128_si32(x32);
}

static CAN_FORCEINLINE int _mm256i_reduce_min_i32(__m256i x)
{
    const __m128i x128 = _mm_min_epi32(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(x, 1));
    const __m128i x64 = _mm_min_epi32(x128, _mm_cvtps_epi32(_mm_movehl_ps(_mm_cvtepi32_ps(x128), _mm_cvtepi32_ps(x128))));
    const __m128i x32 = _mm_min_epi32(x64, _mm_shuffle_epi32(x64, 0x55));
    return _mm_cvtsi128_si32(x32);
}

static CAN_FORCEINLINE int expint(int a, int b)
{
    int _ret = 1;
    for (int i = 0; i < b; ++i)
    {
        _ret = _ret * a;
    }
    return _ret;
}
static void pow_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);

    __m256i _ones_v = _mm256_set1_epi32(1);
    __m256i _zeros_v = _mm256_set1_epi32(0);

    for (int j = 0; j < n8; ++j)
    {
        __m256i iav = _mm256_loadu_si256((__m256i const *)a);
        __m256i ibv = _mm256_loadu_si256((__m256i const *)b);
        __m256i _tmp_v = _mm256_set1_epi32(1);
        ;
        __m256i _tmp_v2;
        int _max = _mm256i_reduce_max_i32(ibv);
        int _min = _mm256i_reduce_min_i32(ibv);
        if (_min <= 0)
        {
            __m256i _flags_v_sign = _mm256_cmpgt_epi32(_zeros_v, ibv);
            for (int i = 0; i < _max; ++i)
            {
                __m256i _flags_v = _mm256_cmpgt_epi32(ibv, _zeros_v);
                _tmp_v2 = _mm256_mullo_epi32(_tmp_v, iav);
                ibv = _mm256_sub_epi32(ibv, _ones_v);
                _tmp_v = _mm256_cvtps_epi32(_mm256_blendv_ps(_mm256_cvtepi32_ps(_tmp_v), _mm256_cvtepi32_ps(_tmp_v2), _mm256_cvtepi32_ps(_flags_v)));
            }
            _tmp_v = _mm256_cvtps_epi32(_mm256_blendv_ps(_mm256_cvtepi32_ps(_tmp_v), _mm256_cvtepi32_ps(_zeros_v), _mm256_cvtepi32_ps(_flags_v_sign)));
        }
        else
        {
            __m256i _v1 = _mm256_set1_epi32(_min);
            ibv = _mm256_sub_epi32(ibv, _v1);
            for (int i = 0; i < _min; ++i)
            {
                _tmp_v = _mm256_mullo_epi32(_tmp_v, iav);
            }
            for (int i = _min; i < _max; ++i)
            {
                __m256i _flags_v = _mm256_cmpgt_epi32(ibv, _zeros_v);
                _tmp_v2 = _mm256_mullo_epi32(_tmp_v, iav);
                ibv = _mm256_sub_epi32(ibv, _ones_v);
                _tmp_v = _mm256_cvtps_epi32(_mm256_blendv_ps(_mm256_cvtepi32_ps(_tmp_v), _mm256_cvtepi32_ps(_tmp_v2), _mm256_cvtepi32_ps(_flags_v)));
            }
        }
        _mm256_storeu_si256((__m256i *)c, _tmp_v);

        c += 8;
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        c[j] = expint(a[j], b[j]);
    }
}

static void logical_and_f32_vec(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256i i_zeros = _mm256_setzero_si256();
    __m256i i_ones = _mm256_set1_epi32(1); // __m256i _mm256_set1_epi32 (int a)
    for (int j = 0; j < n8; ++j)
    {
        __m256i vector_a = _mm256_loadu_si256((__m256i const *)a);
        __m256i vector_b = _mm256_loadu_si256((__m256i const *)b);
        __m256i result_and = _mm256_and_si256(vector_a, vector_b);
        __m256i i_dst = _mm256_cmpeq_epi32(result_and, i_zeros);
        i_dst = _mm256_add_epi32(i_ones, i_dst);
        __m256 f_dst = _mm256_cvtepi32_ps(i_dst);
        _mm256_storeu_ps(c, f_dst);
        a += 8;
        b += 8;
        c += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        int r = (*(a + j)) && (*(b + j));
        if (r)
        {
            c[j] = 1.0f;
        }
        else
        {
            c[j] = 0.0f;
        }
    }
}

static void logical_and_f32_vv(const float *a, const float *b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 f_zeros = _mm256_setzero_ps();
    __m256 f_ones = _mm256_set1_ps(1.0f);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 vector_b = _mm256_loadu_ps(b);
        __m256 result_and = _mm256_and_ps(vector_a, vector_b);
        __m256i i_dst = _mm256_castps_si256(_mm256_cmp_ps(result_and, f_zeros, 8));
        __m256 f_dst = _mm256_cvtepi32_ps(i_dst);
        f_dst = _mm256_add_ps(f_dst, f_ones);
        _mm256_storeu_ps(c, f_dst);

        a += 8;
        b += 8;
        c += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        int r = (*(a + j)) && (*(b + j));
        if (r)
        {
            c[j] = 1.0f;
        }
        else
        {
            c[j] = 0.0f;
        }
    }
}

static void logical_and_f32_vf(const float *a, const float b, float *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256 f_zeros = _mm256_setzero_ps();
    __m256 f_ones = _mm256_set1_ps(1.0f);
    __m256 vector_b = _mm256_set1_ps(b);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 result_and = _mm256_and_ps(vector_a, vector_b);
        __m256i i_dst = _mm256_castps_si256(_mm256_cmp_ps(result_and, f_zeros, 8));
        __m256 f_dst = _mm256_cvtepi32_ps(i_dst);
        f_dst = _mm256_add_ps(f_dst, f_ones);
        _mm256_storeu_ps(c, f_dst);

        a += 8;
        c += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        int r = (*(a + j)) && b;
        if (r)
        {
            c[j] = 1.0f;
        }
        else
        {
            c[j] = 0.0f;
        }
    }
}

static void logical_and_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256i i_zeros = _mm256_setzero_si256();
    __m256i i_ones = _mm256_set1_epi32(1);
    for (int j = 0; j < n8; ++j)
    {
        __m256i vector_a = _mm256_loadu_si256((__m256i const *)a);
        __m256i vector_b = _mm256_loadu_si256((__m256i const *)b);
        __m256i result_and = _mm256_and_si256(vector_a, vector_b);
        __m256i i_dst = _mm256_cmpeq_epi32(result_and, i_zeros);
        i_dst = _mm256_add_epi32(i_ones, i_dst);
        _mm256_storeu_si256((__m256i *)c, i_dst);
        a += 8;
        b += 8;
        c += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        int r = (*(a + j)) && (*(b + j));
        if (r)
        {
            c[j] = 1;
        }
        else
        {
            c[j] = 0;
        }
    }
}

#else // defined(X86_64_SIMD_ON)

operator_vec(add, f32, float, add_fun)
operator_vec(sub, f32, float, sub_fun)
operator_vec(mul, f32, float, mul_fun)
operator_vec(div, f32, float, div_fun)
operator_vec(pow, f32, float, pow_fun)
operator_vec(min, f32, float, min_fun)
operator_vec(max, f32, float, max_fun)
operator_vec(logical_and, f32, float, logical_and_fun)

static void add_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void add_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void sub_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] + b[j];
    }
}

static void sub_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] - b[j];
    }
}

static void mul_f32_vec(const float *a, const float *b, float *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] * b[j];
    }
}

static void mul_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] * b[j];
    }
}

static void div_f32_vec(const float *a, const float *b, float *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] / b[j];
    }
}

static void div_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] / b[j];
    }
}

static void min_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] < b[j] ? a[j] : b[j];
    }
}

static void max_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] > b[j] ? a[j] : b[j];
    }
}

static int expint(int a, int b)
{
    int _ret = 1;
    for (int i = 0; i < b; ++i)
    {
        _ret = _ret * a;
    }
    return _ret;
}

static void pow_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = expint(a[i], b[i]);
    }
}

// static void logical_and_f32_vec(const float *a, const float *b, float *c, int n)
// {
// for (int j = 0; j < n; ++j)
// {
// int r = (*(a + j)) && (*(b + j));
// if (r)
// {
// c[j] = 1.0f;
// }
// else
// {
// c[j] = 0.0f;
// }
// }
// }

static void logical_and_i32_vec(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        int r = (*(a + j)) && (*(b + j));
        if (r)
        {
            c[j] = 1;
        }
        else
        {
            c[j] = 0;
        }
    }
}
#endif // defined(X86_64_SIMD_ON)

static void mul_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] * b[j];
    }
}
static void div_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] / b[j];
    }
}
static void min_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] < b[j] ? a[j] : b[j];
    }
}
static void max_i64_vec(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
    for (int j = 0; j < n; ++j)
    {
        c[j] = a[j] > b[j] ? a[j] : b[j];
    }
}

binary_operator_vec(add, f32, float)
binary_operator_transposition_vec(sub, f32, float)
binary_operator_vec(mul, f32, float)
binary_operator_transposition_vec(div, f32, float)
binary_operator_vec(min, f32, float)
binary_operator_vec(max, f32, float)
binary_operator_vec(pow, f32, float)
binary_operator_vec(logical_and, f32, float)

typedef void (*binary_fun_ptr)(const float *a, int len_a, const float *b, int len_b, float *c, int len_c, int transposition);

template <typename T>
void operator_vec_binary(const T *a, int len_a, const T *b, int len_b, T *c, int len_c, int transposition, binary_fun_ptr f)
{
    (void)len_c;
    int out_len = len_a;
    int inner_len = len_b;
    assert(len_a >= len_b);
    if (inner_len == 1)
    {
        f(a, len_a, b, len_b, c, len_c, transposition);
    }
    else
    {
        int _count = out_len / inner_len;
        for (int i = 0; i < _count; ++i)
        {
            f(a, inner_len, b, inner_len, c, inner_len, transposition);
            a += inner_len;
            c += inner_len;
        }
    }
}

template <typename T>
static void swap_can(T &t1, T &t2)
{
    T tmp = t1;
    t1 = t2;
    t2 = tmp;
}

template <typename T>
int binary_iml(const T *a, const runtime_shape_t &in_a_shape, const T *b, const runtime_shape_t &in_b_shape,
    T *c, const runtime_shape_t &out_shape, binary_fun_ptr f)
{
    const runtime_shape_t *in_a_shape_ptr;
    const runtime_shape_t *in_b_shape_ptr;

    int len_out = (int)compute_size(out_shape);
    int len_a = (int)compute_size(in_a_shape);
    int len_b = (int)compute_size(in_b_shape);
    int transposition = 0;

    if (len_a < len_b)
    {
        swap_can(a, b);
        swap_can(len_a, len_b);
        transposition = 1;
        in_a_shape_ptr = &in_b_shape;
        in_b_shape_ptr = &in_a_shape;
    }
    else
    {
        in_a_shape_ptr = &in_a_shape;
        in_b_shape_ptr = &in_b_shape;
    }
    if (in_b_shape_ptr->size() == 1 || len_a == len_b)
    {
        operator_vec_binary(a, len_a, b, len_b, c, len_out, transposition, f);
    }
    else
    {
        int size_diff = in_a_shape_ptr->size() - in_b_shape_ptr->size();
        int outter_front_size = 1;
        int outter_current_size = 1;
        for (int i = 0; i < size_diff; ++i)
        {
            outter_front_size *= (*in_a_shape_ptr)[i];
        }
        int index = -1;
        for (int i = 0; i < in_b_shape_ptr->size(); ++i)
        {

            if ((*in_b_shape_ptr)[i] == (*in_a_shape_ptr)[i + size_diff])
            {
                outter_current_size *= (*in_b_shape_ptr)[i];
                index = i;
            }
            else
            {
                break;
            }
        }
        if (index == (in_b_shape_ptr->size() - 1)) // [[1, 3, 16, 16], [3, 16, 16]], [[1, 3, 16, 16], [16, 16]], [[1, 3, 16, 16], [16]],
        {
            for (int i = 0; i < outter_front_size; ++i)
            {
                operator_vec_binary(a + outter_current_size * i, outter_current_size, b, outter_current_size, c + outter_current_size * i, len_out, transposition, f);
            }
        }
        else
        {
            int len_a_leave = 1;
            for (int i = index + 1; i < in_b_shape_ptr->size(); ++i)
            {
                len_a_leave *= (*in_a_shape_ptr)[i + size_diff];
            }
            if ((*in_b_shape_ptr)[in_b_shape_ptr->size() - 1] == 1)
            { // [[1, 3, 16, 16],  [3, 1, 1]]，  [[1, 3, 16, 16],  [3, 16, 1]]
                int len_b_leave = 1;
                for (int i = index + 1; i < in_b_shape_ptr->size(); ++i)
                {
                    len_b_leave *= (*in_b_shape_ptr)[i];
                }
                if (len_b_leave != 1)
                {
                    printf("error ... in_b_shape_ptr[i] != 1");
                    return -1;
                }
                for (int j = 0; j < outter_front_size; ++j)
                {
                    for (int i = 0; i < outter_current_size; ++i)
                    {
                        operator_vec_binary(a + len_a_leave * i + j * len_a_leave * outter_current_size,
                            len_a_leave, b + i, 1, c + len_a_leave * i + j * len_a_leave * outter_current_size, 0, transposition, f);
                    }
                }
            }
            else
            {
                int len_b_leave = 1;
                for (int i = index + 1; i < in_b_shape_ptr->size() - 1; ++i)
                {
                    int _data = (*in_b_shape_ptr)[i];
                    if (_data != 1 && (*in_b_shape_ptr)[i + 1] == 1) // 末位非 1 的情况 则不能在 非 1 中间有 1 的情况如 [1, 16, 1, 16]
                    {
                        printf("error ... in_b_shape_ptr[i] == 1\n");
                        return -1;
                    }
                    len_b_leave *= _data;
                }
                len_b_leave *= (*in_b_shape_ptr)[in_b_shape_ptr->size() - 1];
                for (int j = 0; j < outter_front_size; ++j)
                {
                    for (int i = 0; i < outter_current_size; ++i)
                    {
                        operator_vec_binary(a + len_a_leave * i + j * len_a_leave * outter_current_size,
                            len_a_leave, b + i * len_b_leave, len_b_leave, c + len_a_leave * i + j * len_a_leave * outter_current_size, 0, transposition, f);
                    }
                }
            }
        }
    }
    return 0;
}

template <>
result<void> optimized::binary<float>(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, [[maybe_unused]] const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    [[maybe_unused]] const runtime_shape_t &in_b_strides, [[maybe_unused]] const runtime_shape_t &out_shape, [[maybe_unused]] const runtime_shape_t &out_strides,
    [[maybe_unused]] value_range<float> fused_activation, [[maybe_unused]] kernel_context &context) noexcept
{
    int ret = 0;
    if (op == binary_add)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_add_f32_vec);
    }
    else if (op == binary_sub)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_sub_f32_vec);
    }
    else if (op == binary_mul)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_mul_f32_vec);
    }
    else if (op == binary_div)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_div_f32_vec);
    }
    else if (op == binary_min)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_min_f32_vec);
    }
    else if (op == binary_max)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_max_f32_vec);
    }
    else if (op == binary_pow)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_pow_f32_vec);
    }
    else if (op == binary_logical_and)
    {
        ret = binary_iml(input_a, in_a_shape, input_b, in_b_shape, output, out_shape, binary_logical_and_f32_vec);
    }
    else
    {
        return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
            fused_activation, context);
    }
    if (ret)
    {
        return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
            fused_activation, context);
    }
    return ok();
}

template result<void> optimized::binary<int64_t>(binary_op_t op, const int64_t *input_a, const int64_t *input_b, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;
template result<void> optimized::binary<int32_t>(binary_op_t op, const int32_t *input_a, const int32_t *input_b, int32_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept;

template <typename T>
result<void> optimized::binary(binary_op_t op, const T *input_a, const T *input_b, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation, kernel_context &context) noexcept
{
    return cpu::reference::binary(op, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides,
        fused_activation, context);
}
