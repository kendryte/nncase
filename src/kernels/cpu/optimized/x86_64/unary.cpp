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

#if defined(X86_64_SIMD_ON)

#include "avx_mathfun.h"
static void round_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(vector_a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = roundf(a[j]);
    }
}

static void ceil_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(vector_a, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = ceilf(a[j]);
    }
}

static void floor_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_round_ps(vector_a, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = floorf(a[j]);
    }
}

static void sqrt_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sqrt_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = sqrtf(a[j]);
    }
}

static void rsqrt_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 bb = _mm256_rsqrt_ps(aa);
        _mm256_storeu_ps(b, bb);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = 1.0f / sqrtf(a[j]);
    }
}

static void exp_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = exp256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = expf(a[j]);
    }
}

static void log_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = log256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = logf(a[j]);
    }
}

static void cos_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = cos256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = cosf(a[j]);
    }
}

static void sin_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = sin256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = sinf(a[j]);
    }
}

static void negative_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_sub_ps(_mm256_setzero_ps(), vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = -(a[j]);
    }
}

static void logical_not_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256i i_zeros = _mm256_setzero_si256();
    for (int j = 0; j < n8; ++j)
    {
        __m256i vector_a = _mm256_loadu_si256((__m256i const *)a);
        __m256i i_dst_a = _mm256_cmpeq_epi32(vector_a, i_zeros);
        i_dst_a = _mm256_sub_epi32(i_zeros, i_dst_a);
        __m256 f_dst_a = _mm256_cvtepi32_ps(i_dst_a);
        _mm256_storeu_ps(b, f_dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = !a[j];
    }
}

static void abs_f32_vec(const float *a, float *b, int n)
{
    const ALIGN32_BEG int32_t remove_sign_bit_data[8] ALIGN32_END = {
        0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
        0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
    };
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    __m256i remove_sign_bit_flag = _mm256_load_si256((__m256i const *)remove_sign_bit_data);
    for (int j = 0; j < n8; ++j)
    {
        __m256i vector_a = _mm256_loadu_si256((__m256i const *)a);
        __m256i dst_a = _mm256_and_si256(vector_a, remove_sign_bit_flag);
        _mm256_storeu_si256((__m256i *)b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = fabs(a[j]);
    }
}

static void tanh_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vector_a = _mm256_loadu_ps(a);
        __m256 dst_a = tanh256_ps(vector_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = tanhf(a[j]);
    }
}

#ifdef _MSC_VER /* visual c++ */
static CAN_FORCEINLINE float abs_f32(float x)
{
    return fabsf(x);
}
#else /* gcc or icc */
static CAN_FORCEINLINE float abs_f32(float x)
{
    asm(
        "and $0x7FFFFFFF, %0;"
        : "+r"(x)::);
    return x;
}
#endif

static CAN_FORCEINLINE __m256 _mm256_can_acos_ps(__m256 x)
{
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 mtwo = _mm256_set1_ps(-2.0f);
    const __m256 c0 = _mm256_set1_ps(0x1.c86000p-22f); //  4.25032340e-7
    const __m256 c1 = _mm256_set1_ps(-0x1.0258fap-19f); // -1.92483935e-6
    const __m256 c2 = _mm256_set1_ps(0x1.90c5c4p-18f); //  5.97197595e-6
    const __m256 c3 = _mm256_set1_ps(-0x1.55668cp-19f); // -2.54363249e-6
    const __m256 c4 = _mm256_set1_ps(0x1.c3f78ap-16f); //  2.69393295e-5
    const __m256 c5 = _mm256_set1_ps(0x1.e8f446p-14f); //  1.16575764e-4
    const __m256 c6 = _mm256_set1_ps(0x1.6df072p-11f); //  6.97973708e-4
    const __m256 c7 = _mm256_set1_ps(0x1.3332a6p-8f); //  4.68746712e-3
    const __m256 c8 = _mm256_set1_ps(0x1.555550p-5f); //  4.16666567e-2
    const __m256 pi0 = _mm256_set1_ps(0x1.ddcb02p+0f); //  1.86637890e+0
    const __m256 pi1 = _mm256_set1_ps(0x1.aee9d6p+0f); //  1.68325555e+0
    __m256 s, r, t, m;

    s = two;
    t = mtwo;
    m = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    t = _mm256_blendv_ps(t, s, m);
    t = _mm256_fmadd_ps(x, t, s);
    s = _mm256_sqrt_ps(t);
    r = c0;
    r = _mm256_fmadd_ps(r, t, c1);
    r = _mm256_fmadd_ps(r, t, c2);
    r = _mm256_fmadd_ps(r, t, c3);
    r = _mm256_fmadd_ps(r, t, c4);
    r = _mm256_fmadd_ps(r, t, c5);
    r = _mm256_fmadd_ps(r, t, c6);
    r = _mm256_fmadd_ps(r, t, c7);
    r = _mm256_fmadd_ps(r, t, c8);
    r = _mm256_mul_ps(r, t);
    r = _mm256_fmadd_ps(r, s, s);
    t = _mm256_sub_ps(zero, r);
    t = _mm256_fmadd_ps(pi0, pi1, t);
    r = _mm256_blendv_ps(r, t, m);
    return r;
}

//t > 0.921875f
static CAN_FORCEINLINE __m256 erf_core_ps1(__m256 a, __m256 t, __m256 s, __m256 r0, __m256 r1, __m256 r2,
    __m256 r3, __m256 r4, __m256 r5, __m256 r6, __m256i n_flag)
{
    __m256 r = _mm256_fmadd_ps(r0, t, r1);
    __m256 u = _mm256_fmadd_ps(r2, t, r3);
    r = _mm256_fmadd_ps(r, s, u);
    r = _mm256_fmadd_ps(r, t, r4);
    r = _mm256_fmadd_ps(r, t, r5);
    r = _mm256_fmadd_ps(r, t, r6);
    r = _mm256_fmadd_ps(r, t, t);
    __m256 _zeros = _mm256_setzero_ps();
    __m256 _ones = _mm256_set1_ps(1.0f);
    __m256 minus_r = _mm256_sub_ps(_zeros, r);
    r = exp256_ps(minus_r);
    r = _mm256_sub_ps(_ones, r);

    __m256i sign_flag = _mm256_andnot_si256(n_flag, _mm256_castps_si256(a));
    __m256i pr = _mm256_and_si256(n_flag, _mm256_castps_si256(r));
    r = _mm256_castsi256_ps(_mm256_or_si256(sign_flag, pr));
    return r;
}

// t <= 0.921875f
static CAN_FORCEINLINE __m256 erf_core_ps2(__m256 a, __m256 s, __m256 r1, __m256 r2,
    __m256 r3, __m256 r4, __m256 r5, __m256 r6)
{
    __m256 r = _mm256_fmadd_ps(r1, s, r2);
    r = _mm256_fmadd_ps(r, s, r3);
    r = _mm256_fmadd_ps(r, s, r4);
    r = _mm256_fmadd_ps(r, s, r5);
    r = _mm256_fmadd_ps(r, s, r6);
    r = _mm256_fmadd_ps(r, a, a);
    return r;
}

static void erf_f32_vec(const float *a, float *b, int n)
{
    const float erf_const_data[] = { -0x1.3a1a82p-11f, 0x1.473f48p-08f, -0x1.b68bd2p-06f,
        0x1.ce1a46p-04f, -0x1.8126e0p-02f, 0x1.06eba6p-03f };
    __m256 r1 = _mm256_broadcast_ss(erf_const_data);
    __m256 r2 = _mm256_broadcast_ss(erf_const_data + 1);
    __m256 r3 = _mm256_broadcast_ss(erf_const_data + 2);
    __m256 r4 = _mm256_broadcast_ss(erf_const_data + 3);
    __m256 r5 = _mm256_broadcast_ss(erf_const_data + 4);
    __m256 r6 = _mm256_broadcast_ss(erf_const_data + 5);

    /////////////////////////////
    // if t > 0.921875f
    const __m256 c0 = _mm256_set1_ps(0x1.222900p-16f);
    const __m256 c1 = _mm256_set1_ps(-0x1.91d2ccp-12f);
    const __m256 c2 = _mm256_set1_ps(0x1.fd1336p-09f);
    const __m256 c3 = _mm256_set1_ps(-0x1.8d6300p-06f);
    const __m256 c4 = _mm256_set1_ps(0x1.b55cb0p-4f);
    const __m256 c5 = _mm256_set1_ps(0x1.450aa0p-1f);
    const __m256 c6 = _mm256_set1_ps(0x1.079d0cp-3f);
    const __m256 c7 = _mm256_set1_ps(0.921875f);
    /////////////////////////////

    __m256i n_flag = _mm256_set1_epi32(0x7fffffff);

    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 s = _mm256_mul_ps(aa, aa); // s
        __m256 t = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(aa), n_flag));

        __m256 ret1 = erf_core_ps1(aa, t, s, c0, c1, c2, c3, c4, c5, c6, n_flag);
        __m256 ret2 = erf_core_ps2(aa, s, r1, r2, r3, r4, r5, r6);

        __m256 _flag = _mm256_cmp_ps(t, c7, _CMP_LT_OQ);

        ret1 = _mm256_blendv_ps(ret1, ret2, _flag);
        _mm256_storeu_ps(b, ret1);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = erff(a[j]);
    }
}

static void sign_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 aa = _mm256_loadu_ps(a);
        __m256 b1 = _mm256_cmp_ps(_mm256_setzero_ps(), aa, _CMP_LT_OQ);
        __m256 b2 = _mm256_cmp_ps(aa, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256i ib1 = _mm256_castps_si256(b1);
        __m256i ib2 = _mm256_castps_si256(b2);
        __m256i ret = _mm256_sub_epi32(ib2, ib1);

        __m256 kbb = _mm256_cvtepi32_ps(ret);
        _mm256_storeu_ps(b, kbb);

        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = (0.f < a[j]) - (a[j] < 0.f);
    }
}

static void acos_f32_vec(const float *a, float *b, int n)
{
    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 vecotr_a = _mm256_loadu_ps(a);
        __m256 dst_a = _mm256_can_acos_ps(vecotr_a);
        _mm256_storeu_ps(b, dst_a);
        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = acosf(a[j]);
    }
}

CAN_FORCEINLINE __m256 asinf_core_ps(__m256 a, __m256 r0, __m256 r1, __m256 r2, __m256 r3, __m256 r4)
{
    __m256 ss = _mm256_mul_ps(a, a); // s = a * a;
    __m256 r = r0;
    r = _mm256_fmadd_ps(r, ss, r1); //r = fmaf(r, s, 0x1.29a5cep-6f); // 1.81669723e-23
    r = _mm256_fmadd_ps(r, ss, r2);
    r = _mm256_fmadd_ps(r, ss, r3);
    r = _mm256_fmadd_ps(r, ss, r4);
    r = _mm256_mul_ps(r, ss);
    r = _mm256_fmadd_ps(r, a, a);
    return r;
}

CAN_FORCEINLINE __m256 asinf_core2_ps(__m256 a, __m256 r0, __m256 r1, __m256 r2, __m256 r3, __m256 r4, __m256 one_256, __m256 half_one_256, __m256 half_pi_256,
    __m256i abs_flag, __m256i sign_flag)
{
    __m256 s; // = a;

    ////////////////////
    // 获取符号位
    __m256i isign_flag = _mm256_and_si256(_mm256_castps_si256(a), sign_flag);
    __m256i _xv = _mm256_and_si256(_mm256_castps_si256(a), abs_flag);
    s = _mm256_castsi256_ps(_xv);
    ////////////////////

    ////////////////////////////
    //  before
    s = _mm256_sub_ps(one_256, s); // 1 - x
    s = _mm256_mul_ps(half_one_256, s); // (1 - x) / 2
    s = _mm256_sqrt_ps(s);
    /////////////////////////////

    __m256 ss = _mm256_mul_ps(s, s); // s = a * a;
    __m256 r = r0;
    r = _mm256_fmadd_ps(r, ss, r1); //r = fmaf(r, s, 0x1.29a5cep-6f); // 1.81669723e-23
    r = _mm256_fmadd_ps(r, ss, r2);
    r = _mm256_fmadd_ps(r, ss, r3);
    r = _mm256_fmadd_ps(r, ss, r4);
    r = _mm256_mul_ps(r, ss);
    r = _mm256_fmadd_ps(r, s, s);

    ////////////////////////////
    //  after
    s = _mm256_div_ps(r, half_one_256); // 2 * asinf_core(x)
    s = _mm256_sub_ps(half_pi_256, s); // pi / 2 - 2 * asinf_core(x)
    /////////////////////////////

    ////////////////////
    // 恢复符号位
    s = _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(s), isign_flag));
    return s;
}

void asinf_f32_vec(const float *a, float *b, int n)
{
    const float pi = 3.1415926f;
    const float __init__data[] = { 0x1.a7f260p-5f, 0x1.29a5cep-6f, 0x1.7f0842p-5f, 0x1.329256p-4f, 0x1.555728p-3f, 1.0f, 0.5f, pi / 2 };
    __m256 r0 = _mm256_broadcast_ss(__init__data);
    __m256 r1 = _mm256_broadcast_ss(__init__data + 1);
    __m256 r2 = _mm256_broadcast_ss(__init__data + 2);
    __m256 r3 = _mm256_broadcast_ss(__init__data + 3);
    __m256 r4 = _mm256_broadcast_ss(__init__data + 4);

    __m256 one_256 = _mm256_broadcast_ss(__init__data + 5);
    __m256 half_one_256 = _mm256_broadcast_ss(__init__data + 6);
    __m256 half_pi_256 = _mm256_broadcast_ss(__init__data + 7);

    const ALIGN32_BEG int32_t x1[8] ALIGN32_END = {
        0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
        0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
    };
    const ALIGN32_BEG uint32_t x2[8] ALIGN32_END = {
        0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0x80000000, 0x80000000
    };

    __m256i abs_flag = _mm256_load_si256((__m256i const *)x1);
    __m256i sign_flag = _mm256_load_si256((__m256i const *)x2);

    int n8 = (n >> 3);
    int n8_left = n & (8 - 1);
    for (int j = 0; j < n8; ++j)
    {
        __m256 s = _mm256_loadu_ps(a);
        __m256 s1 = asinf_core_ps(s, r0, r1, r2, r3, r4);
        ////////////
        // fabsf 是否大于 0.5f
        /////////////
        __m256 abs_s = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(s), abs_flag));
        ////__m256 _mm256_cmp_ps(__m256 a, __m256 b, const int imm8);
        __m256 flags_half_2 = _mm256_cmp_ps(abs_s, half_one_256, _CMP_NLT_UQ);

        __m256 flags_half_2_1 = _mm256_cmp_ps(half_one_256, abs_s, _CMP_NLT_UQ);

        __m256 s2 = asinf_core2_ps(s, r0, r1, r2, r3, r4, one_256, half_one_256, half_pi_256, abs_flag, sign_flag);

        s1 = _mm256_and_ps(s1, flags_half_2_1);
        s2 = _mm256_and_ps(s2, flags_half_2);
        s2 = _mm256_or_ps(s1, s2);
        _mm256_storeu_ps(b, s2);

        a += 8;
        b += 8;
    }
    for (int j = 0; j < n8_left; ++j)
    {
        b[j] = asinf(a[j]);
    }
}
#else // X86_64_SIMD_ON

static void round_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = roundf(a[j]);
    }
}

static void ceil_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = ceilf(a[j]);
    }
}

static void floor_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = floorf(a[j]);
    }
}

static void sqrt_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = sqrtf(a[j]);
    }
}

static void rsqrt_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = 1.0f / sqrtf(a[j]);
    }
}

static void exp_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = expf(a[j]);
    }
}

static void log_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = logf(a[j]);
    }
}

static void cos_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = cosf(a[j]);
    }
}

static void sin_f32_vec(const float *a, float *b, int n)
{

    for (int j = 0; j < n; ++j)
    {
        b[j] = sinf(a[j]);
    }
}

static void negative_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = -(a[j]);
    }
}

static void logical_not_f32_vec(const float *a, float *b, int n)
{

    for (int j = 0; j < n; ++j)
    {
        b[j] = !a[j];
    }
}

static void abs_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = fabs(a[j]);
    }
}

static void tanh_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = tanhf(a[j]);
    }
}

static void erf_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = erff(a[j]);
    }
}

static void sign_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = (0.f < a[j]) - (a[j] < 0.f);
    }
}

static void acos_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = acosf(a[j]);
    }
}

static void asinf_f32_vec(const float *a, float *b, int n)
{
    for (int j = 0; j < n; ++j)
    {
        b[j] = asinf(a[j]);
    }
}
#endif // X86_64_SIMD_ON

result<void> optimized::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    result<void> ret_value = ok();
    int len = (int)compute_size(shape);

    if (op == unary_round)
    {
        round_f32_vec(input, output, len);
    }
    else if (op == unary_ceil)
    {
        ceil_f32_vec(input, output, len);
    }
    else if (op == unary_floor)
    {
        floor_f32_vec(input, output, len);
    }
    else if (op == unary_sqrt)
    {
        sqrt_f32_vec(input, output, len);
    }
    else if (op == unary_rsqrt)
    {
        rsqrt_f32_vec(input, output, len);
    }
    else if (op == unary_exp)
    {
        exp_f32_vec(input, output, len);
    }
    else if (op == unary_log)
    {
        log_f32_vec(input, output, len);
    }
    else if (op == unary_cos)
    {
        cos_f32_vec(input, output, len);
    }
    else if (op == unary_sin)
    {
        sin_f32_vec(input, output, len);
    }
    else if (op == unary_neg)
    {
        negative_f32_vec(input, output, len);
    }
    else if (op == unary_abs)
    {
        abs_f32_vec(input, output, len);
    }
    else if (op == unary_logical_not)
    {
        logical_not_f32_vec(input, output, len);
    }
    else if (op == unary_tanh)
    {
        tanh_f32_vec(input, output, len);
    }
    else if (op == unary_erf)
    {
        erf_f32_vec(input, output, len);
    }
    else if (op == unary_sign)
    {
        sign_f32_vec(input, output, len);
    }
    else if (op == unary_acos)
    {
        acos_f32_vec(input, output, len);
    }
    else if (op == unary_asin)
    {
        asinf_f32_vec(input, output, len);
    }
    else
    {
        ret_value = cpu::reference::unary(op, input, output, shape, in_strides, out_strides, context);
    }
    return ret_value;
}
