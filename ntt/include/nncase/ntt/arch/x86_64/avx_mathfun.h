/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef AVX_MATHFUN_H
#define AVX_MATHFUN_H

#include "x86_usability.h"
#include <emmintrin.h>
#include <immintrin.h>

/* yes I know, the top of this file is quite ugly */

#ifdef _MSC_VER /* visual c++ */
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#else /* gcc or icc */
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#endif

#define _PI32AVX_CONST(Name, Val)                                              \
    static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = {Val, Val,   \
                                                                   Val, Val}

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                                \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = {            \
        Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val)                                              \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = {           \
        Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val)                                     \
    static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = {             \
        Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);

#ifndef __AVX2__
typedef union imm_xmm_union {
    __m256i imm;
    __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)                                    \
    {                                                                          \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END;                               \
        u.imm = imm_;                                                          \
        xmm0_ = u.xmm[0];                                                      \
        xmm1_ = u.xmm[1];                                                      \
    }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)                                    \
    {                                                                          \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END;                               \
        u.xmm[0] = xmm0_;                                                      \
        u.xmm[1] = xmm1_;                                                      \
        imm_ = u.imm;                                                          \
    }

#define AVX2_BITOP_USING_SSE2(fn)                                              \
    static inline __m256i _mm256_comp_##fn(__m256i x, int a) {                 \
        /* use SSE2 instruction to perform the bitop AVX2 */                   \
        __m128i x1, x2;                                                        \
        __m256i ret;                                                           \
        COPY_IMM_TO_XMM(x, x1, x2);                                            \
        x1 = _mm_##fn(x1, a);                                                  \
        x2 = _mm_##fn(x2, a);                                                  \
        COPY_XMM_TO_IMM(x1, x2, ret);                                          \
        return (ret);                                                          \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                              \
    static inline __m256i _mm256_comp_##fn(__m256i x, __m256i y) {             \
        /* use SSE2 instructions to perform the AVX2 integer operation */      \
        __m128i x1, x2;                                                        \
        __m128i y1, y2;                                                        \
        __m256i ret;                                                           \
        COPY_IMM_TO_XMM(x, x1, x2);                                            \
        COPY_IMM_TO_XMM(y, y1, y2);                                            \
        x1 = _mm_##fn(x1, y1);                                                 \
        x2 = _mm_##fn(x2, y2);                                                 \
        COPY_XMM_TO_IMM(x1, x2, ret);                                          \
        return (ret);                                                          \
    }
#else
#define AVX2_BITOP_USING_SSE2(fn)                                              \
    static inline __m256i _mm256_comp_##fn(__m256i x, int a) {                 \
        return _mm256_##fn(x, a);                                              \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                              \
    static inline __m256i _mm256_comp_##fn(__m256i x, __m256i y) {             \
        return _mm256_##fn(x, y);                                              \
    }
#endif

AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)

// Replace 256 bit operations with 128 bit ones when AVX2 is disabled
#ifndef __AVX2__
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
#endif

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
static inline __m256 log256_ps(__m256 x) {
    __m256i imm0;
    __m256 one = *(__m256 *)_ps256_1;

    //__m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(
        x, *(__m256 *)_ps256_min_norm_pos); /* cut off denormalized stuff */

    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(__m256 *)_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(__m256 *)_ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_comp_sub_epi32(imm0, *(__m256i *)_pi32_256_0x7f);
    __m256 e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    //__m256 mask = _mm256_cmplt_ps(x, *(__m256*)_ps256_cephes_SQRTHF);
    __m256 mask = _mm256_cmp_ps(x, *(__m256 *)_ps256_cephes_SQRTHF, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);

    __m256 y = *(__m256 *)_ps256_cephes_log_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p5);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p6);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p7);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    y = _mm256_comp_fmadd_ps(e, *(__m256 *)_ps256_cephes_log_q1, y);

    // y = -z * 0.5 + y
    y = _mm256_comp_fnmadd_ps(z, *(__m256 *)_ps256_0p5, y);

    x = _mm256_add_ps(x, y);
    x = _mm256_comp_fmadd_ps(e, *(__m256 *)_ps256_cephes_log_q2, x);
    y = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    return y;
}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

static inline __m256 exp256_ps(__m256 x) {
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256 *)_ps256_1;

    x = _mm256_min_ps(x, *(__m256 *)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256 *)_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_comp_fmadd_ps(x, *(__m256 *)_ps256_cephes_LOG2EF,
                              *(__m256 *)_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    // imm0 = _mm256_cvttps_epi32(fx);
    // tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, subtract 1 */
    //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    // x = x - fx * exp_C1
    x = _mm256_comp_fnmadd_ps(fx, *(__m256 *)_ps256_cephes_exp_C1, x);
    // x = x - fx * exp_C2
    x = _mm256_comp_fnmadd_ps(fx, *(__m256 *)_ps256_cephes_exp_C2, x);

    tmp = _mm256_mul_ps(x, x);

    __m256 y = *(__m256 *)_ps256_cephes_exp_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_exp_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_exp_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_exp_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_exp_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256 *)_ps256_cephes_exp_p5);
    y = _mm256_comp_fmadd_ps(y, tmp, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_comp_add_epi32(imm0, *(__m256i *)_pi32_256_0x7f);
    imm0 = _mm256_comp_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

_PS256_CONST(tanh_hi, 9.0f);
_PS256_CONST(tanh_lo, -9.0f);

_PS256_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS256_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS256_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS256_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS256_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS256_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS256_CONST(cephes_tanh_p6, 4.89352455891786E-03f);

_PS256_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS256_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS256_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

// an approximation of tanh
static inline __m256 tanh256_ps(const __m256 x) {
    __m256 value = x;
    value = _mm256_max_ps(*(__m256 *)_ps256_tanh_lo, value);
    value = _mm256_min_ps(*(__m256 *)_ps256_tanh_hi, value);

    __m256 value_squared = _mm256_mul_ps(value, value);

    __m256 p;
    p = _mm256_comp_fmadd_ps(value_squared, *(__m256 *)_ps256_cephes_tanh_p0,
                             *(__m256 *)_ps256_cephes_tanh_p1);
    p = _mm256_comp_fmadd_ps(p, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p2);
    p = _mm256_comp_fmadd_ps(p, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p3);
    p = _mm256_comp_fmadd_ps(p, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p4);
    p = _mm256_comp_fmadd_ps(p, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p5);
    p = _mm256_comp_fmadd_ps(p, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p6);
    p = _mm256_mul_ps(p, value);

    __m256 q;
    q = _mm256_comp_fmadd_ps(value_squared, *(__m256 *)_ps256_cephes_tanh_p7,
                             *(__m256 *)_ps256_cephes_tanh_p8);
    q = _mm256_comp_fmadd_ps(q, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p9);
    q = _mm256_comp_fmadd_ps(q, value_squared,
                             *(__m256 *)_ps256_cephes_tanh_p6);

    __m256 dst = _mm256_div_ps(p, q);
    return dst;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625f);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1, 8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0, 2.443315711809948E-005f);
_PS256_CONST(coscof_p1, -1.388731625493765E-003f);
_PS256_CONST(coscof_p2, 4.166664568298827E-002f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
static inline __m256 sin256_ps(__m256 x) { // any x
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    __m256i imm0, imm2;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
#endif

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256 *)_ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, *(__m256 *)_ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256 *)_ps256_cephes_FOPI);

    /*
      Here we start a series of integer operations, which are in the
      realm of AVX2.
      If we don't have AVX, let's perform them using SSE2 directives
    */

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i *)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i *)_pi32_256_0);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i *)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i *)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm0_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    __m256 swap_sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256 *)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256 *)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256 *)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m256 *)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);

    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256 *)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256 *)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256 *)_ps256_sincof_p0;
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256 *)_ps256_sincof_p1);
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256 *)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_comp_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static inline __m256 cos256_ps(__m256 x) { // any x
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    __m256i imm0, imm2;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
#endif

    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256 *)_ps256_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256 *)_ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i *)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_comp_sub_epi32(imm2, *(__m256i *)_pi32_256_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, *(__m256i *)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i *)_pi32_256_0);
#else

    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i *)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i *)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm2_1 = _mm_sub_epi32(imm2_1, *(__m128i *)_pi32avx_2);
    imm2_2 = _mm_sub_epi32(imm2_2, *(__m128i *)_pi32avx_2);

    imm0_1 = _mm_andnot_si128(imm2_1, *(__m128i *)_pi32avx_4);
    imm0_2 = _mm_andnot_si128(imm2_2, *(__m128i *)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    __m256 sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256 *)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256 *)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256 *)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m256 *)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);

    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256 *)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256 *)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256 *)_ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256 *)_ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256 *)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could
   replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static inline void sincos256_ps(__m256 x, __m256 *s, __m256 *c) {
    __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    __m256i imm0, imm2, imm4;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
    __m128i imm4_1, imm4_2;
#endif

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256 *)_ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256 *)_ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256 *)_ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i *)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    //__m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, *(__m256i *)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i *)_pi32_256_0);
    //__m256 poly_mask = _mm256_castsi256_ps(imm2);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i *)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i *)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm4_1 = imm2_1;
    imm4_2 = imm2_2;

    imm0_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i *)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i *)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256 *)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256 *)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256 *)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

#ifdef __AVX2__
    imm4 = _mm256_comp_sub_epi32(imm4, *(__m256i *)_pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(__m256i *)_pi32_256_4);
    imm4 = _mm256_comp_slli_epi32(imm4, 29);
#else
    imm4_1 = _mm_sub_epi32(imm4_1, *(__m128i *)_pi32avx_2);
    imm4_2 = _mm_sub_epi32(imm4_2, *(__m128i *)_pi32avx_2);

    imm4_1 = _mm_andnot_si128(imm4_1, *(__m128i *)_pi32avx_4);
    imm4_2 = _mm_andnot_si128(imm4_2, *(__m128i *)_pi32avx_4);

    imm4_1 = _mm_slli_epi32(imm4_1, 29);
    imm4_2 = _mm_slli_epi32(imm4_2, 29);

    COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
#endif

    __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m256 z = _mm256_mul_ps(x, x);
    y = *(__m256 *)_ps256_coscof_p0;

    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256 *)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256 *)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256 *)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256 *)_ps256_sincof_p0;
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256 *)_ps256_sincof_p1);
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256 *)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_comp_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);

    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

static inline __m256 tan256_ps(__m256 x) {
    __m256 ysin, ycos;
    __m256 eps = _mm256_set1_ps(1E-8f);
    sincos256_ps(x, &ysin, &ycos);
    __m256 mask = _mm256_cmp_ps(ycos, _mm256_setzero_ps(), _CMP_EQ_OS);
    __m256 _tmp = _mm256_and_ps(eps, mask);
    ycos = _mm256_add_ps(ycos, _tmp);
    __m256 ytan = _mm256_div_ps(ysin, ycos);
    return ytan;
}

static inline __m256 pow256_ps(__m256 a, __m256 b) {
    // pow(x, m) = exp(m * log(x))
    return exp256_ps(_mm256_mul_ps(b, log256_ps(a)));
}

static inline __m256 asin256_ps(__m256 x) {
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_half_one = _mm256_set1_ps(0.5f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_a4 = _mm256_set1_ps(0.023994016f);
    const __m256 magic_a5 = _mm256_set1_ps(0.042417344f);
    const __m256 magic_a2 = _mm256_set1_ps(0.07494697f);
    const __m256 magic_a3 = _mm256_set1_ps(0.045520633f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(0.166667819f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);
    const __m256 magic_three = _mm256_set1_ps(3.0f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __m256 is_small_input = _mm256_cmp_ps(absolute, magic_half_one, _CMP_LE_OQ);

    // is_big_input = (is_small_input ? 0.0f : 1.0f);
    __m256 is_big_input = _mm256_andnot_ps(is_small_input, magic_one);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m256 big_input_approx = _mm256_sqrt_ps(
        _mm256_mul_ps(magic_half_one, _mm256_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m256 input_approx =
        _mm256_or_ps(_mm256_and_ps(is_small_input, absolute),
                     _mm256_andnot_ps(is_small_input, big_input_approx));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx =
        _mm256_mul_ps(square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m256 output_approx = _mm256_comp_fmadd_ps(
        square_of_input_approx,
        _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                             _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                                                  magic_a5, magic_a3),
                             magic_a1),
        _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                             _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                                                  magic_a4, magic_a2),
                             magic_a0));

    // TODO: Need more explanations.
    // x1 = ((0.5 * PI) * is_big_input);
    // x2 = (output_approx * input_approx);
    // x3 = (-(3.0f * is_big_input) + 1.0f);
    // final_approx = ((x2 * x3) + x1);
    __m256 final_approx = _mm256_comp_fmadd_ps(
        _mm256_mul_ps(output_approx, input_approx),
        _mm256_comp_fnmadd_ps(magic_three, is_big_input, magic_one),
        _mm256_mul_ps(magic_half_pi, is_big_input));

    // return (final_approx || negative_mask);
    return _mm256_or_ps(final_approx, negative_mask);
}

static inline __m256 acos256_ps(__m256 x) {
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_zero = _mm256_set1_ps(0.0f);
    const __m256 magic_half_one = _mm256_set1_ps(0.5f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_a4 = _mm256_set1_ps(0.023994016f);
    const __m256 magic_a5 = _mm256_set1_ps(0.042417344f);
    const __m256 magic_a2 = _mm256_set1_ps(0.07494697f);
    const __m256 magic_a3 = _mm256_set1_ps(0.045520633f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(0.166667819f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.57079632679489661923f);
    const __m256 magic_pi =
        _mm256_set1_ps(3.14159265358979323846264338327950288f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __m256 is_small_input = _mm256_cmp_ps(absolute, magic_half_one, _CMP_LE_OQ);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m256 big_input_approx = _mm256_sqrt_ps(
        _mm256_mul_ps(magic_half_one, _mm256_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m256 input_approx =
        _mm256_or_ps(_mm256_and_ps(is_small_input, absolute),
                     _mm256_andnot_ps(is_small_input, big_input_approx));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx =
        _mm256_mul_ps(square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m256 output_approx = _mm256_comp_fmadd_ps(
        square_of_input_approx,
        _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                             _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                                                  magic_a5, magic_a3),
                             magic_a1),
        _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                             _mm256_comp_fmadd_ps(fourth_power_of_input_approx,
                                                  magic_a4, magic_a2),
                             magic_a0));

    // TODO: Need more explanations.
    // x1 = (output_approx * input_approx);
    __m256 x1 = _mm256_mul_ps(output_approx, input_approx);

    // TODO: Need more explanations.
    // small_final_approx = ((0.5 * PI) - (x1 | negative_mask));
    __m256 small_final_approx =
        _mm256_sub_ps(magic_half_pi, _mm256_or_ps(x1, negative_mask));

    // TODO: Need more explanations.
    // big_final_approx = (((x < 0.0f) & PI) + ((x1 * 2) | negative_mask));
    __m256 big_final_approx = _mm256_add_ps(
        _mm256_and_ps(_mm256_cmp_ps(x, magic_zero, _CMP_LT_OQ), magic_pi),
        _mm256_or_ps(_mm256_add_ps(x1, x1), negative_mask));

    // return (is_small_input ? small_final_approx : big_final_approx);
    return _mm256_or_ps(_mm256_and_ps(is_small_input, small_final_approx),
                        _mm256_andnot_ps(is_small_input, big_final_approx));
}

static inline __m256 atan256_ps(__m256 x) {
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_negative_one = _mm256_set1_ps(-1.0f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(-0.33333072f);
    const __m256 magic_a2 = _mm256_set1_ps(0.1999262f);
    const __m256 magic_a3 = _mm256_set1_ps(-0.14203644f);
    const __m256 magic_a4 = _mm256_set1_ps(0.10640934f);
    const __m256 magic_a5 = _mm256_set1_ps(-0.07504295f);
    const __m256 magic_a6 = _mm256_set1_ps(0.04269152f);
    const __m256 magic_a7 = _mm256_set1_ps(-0.01606863f);
    const __m256 magic_a8 = _mm256_set1_ps(0.0028498897f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (1.0f < absolute);
    __m256 is_small_input = _mm256_cmp_ps(magic_one, absolute, _CMP_LT_OQ);

    // x1 = (is_small_input ? -1.0f : absolute);
    // x2 = (is_small_input ? absolute : 1.0f)
    // input_approx = x1 / x2;
    __m256 input_approx = _mm256_div_ps(
        _mm256_or_ps(_mm256_and_ps(is_small_input, magic_negative_one),
                     _mm256_andnot_ps(is_small_input, absolute)),
        _mm256_or_ps(_mm256_and_ps(is_small_input, absolute),
                     _mm256_andnot_ps(is_small_input, magic_one)));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx =
        _mm256_mul_ps(square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a7) + magic_a5);
    // x2 = ((fourth_power_of_input_approx * magic_a8) + magic_a6);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a3);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a4);
    // x5 = ((fourth_power_of_input_approx * x3) + magic_a1);
    // x6 = ((fourth_power_of_input_approx * x4) + magic_a2);
    // x7 = ((fourth_power_of_input_approx * x6) + magic_a0);
    // output_approx = ((square_of_input_approx * x5) + x7);
    __m256 output_approx = _mm256_comp_fmadd_ps(
        square_of_input_approx,
        _mm256_comp_fmadd_ps(
            fourth_power_of_input_approx,
            _mm256_comp_fmadd_ps(
                fourth_power_of_input_approx,
                _mm256_comp_fmadd_ps(fourth_power_of_input_approx, magic_a7,
                                     magic_a5),
                magic_a3),
            magic_a1),
        _mm256_comp_fmadd_ps(
            fourth_power_of_input_approx,
            _mm256_comp_fmadd_ps(
                fourth_power_of_input_approx,
                _mm256_comp_fmadd_ps(
                    fourth_power_of_input_approx,
                    _mm256_comp_fmadd_ps(fourth_power_of_input_approx, magic_a8,
                                         magic_a6),
                    magic_a4),
                magic_a2),
            magic_a0));

    // TODO: Need more explanations.
    // x1 = (output_approx * input_approx);
    // if (is_small_input) x1 += (0.5 * PI);
    // return (negative_mask ? -x1 : x1);
    return _mm256_or_ps(
        _mm256_add_ps(_mm256_mul_ps(output_approx, input_approx),
                      _mm256_and_ps(is_small_input, magic_half_pi)),
        negative_mask);
}

struct sv_erff_data {
    float erf[513];
    float scale[513];
};

/* Lookup table used in SVE erff.
   For each possible rounded input r (multiples of 1/128), between
   r = 0.0 and r = 4.0 (513 values):
   - __erff_data.erf contains the values of erf(r),
   - __erff_data.scale contains the values of 2/sqrt(pi)*exp(-r^2).
   Note that indices 0 and 1 are never hit by the algorithm, since lookup is
   performed only for x >= 1/64-1/512.  */
const struct sv_erff_data __sv_erff_data = {
    .erf =
        {
            0x0.000000p+0, 0x0.000000p+0, 0x1.20d770p-6, 0x1.b137e0p-6,
            0x1.20c564p-5, 0x1.68e5d4p-5, 0x1.b0fafep-5, 0x1.f902a8p-5,
            0x1.207d48p-4, 0x1.44703ep-4, 0x1.68591ap-4, 0x1.8c36bep-4,
            0x1.b00812p-4, 0x1.d3cbf8p-4, 0x1.f7815ap-4, 0x1.0d9390p-3,
            0x1.1f5e1ap-3, 0x1.311fc2p-3, 0x1.42d7fcp-3, 0x1.548642p-3,
            0x1.662a0cp-3, 0x1.77c2d2p-3, 0x1.895010p-3, 0x1.9ad142p-3,
            0x1.ac45e4p-3, 0x1.bdad72p-3, 0x1.cf076ep-3, 0x1.e05354p-3,
            0x1.f190aap-3, 0x1.015f78p-2, 0x1.09eed6p-2, 0x1.127632p-2,
            0x1.1af54ep-2, 0x1.236bf0p-2, 0x1.2bd9dcp-2, 0x1.343ed6p-2,
            0x1.3c9aa8p-2, 0x1.44ed18p-2, 0x1.4d35f0p-2, 0x1.5574f4p-2,
            0x1.5da9f4p-2, 0x1.65d4b8p-2, 0x1.6df50ap-2, 0x1.760abap-2,
            0x1.7e1594p-2, 0x1.861566p-2, 0x1.8e0a02p-2, 0x1.95f336p-2,
            0x1.9dd0d2p-2, 0x1.a5a2acp-2, 0x1.ad6896p-2, 0x1.b52264p-2,
            0x1.bccfecp-2, 0x1.c47104p-2, 0x1.cc0584p-2, 0x1.d38d44p-2,
            0x1.db081cp-2, 0x1.e275eap-2, 0x1.e9d68ap-2, 0x1.f129d4p-2,
            0x1.f86faap-2, 0x1.ffa7eap-2, 0x1.03693ap-1, 0x1.06f794p-1,
            0x1.0a7ef6p-1, 0x1.0dff50p-1, 0x1.117894p-1, 0x1.14eab4p-1,
            0x1.1855a6p-1, 0x1.1bb95cp-1, 0x1.1f15ccp-1, 0x1.226ae8p-1,
            0x1.25b8a8p-1, 0x1.28ff02p-1, 0x1.2c3decp-1, 0x1.2f755cp-1,
            0x1.32a54cp-1, 0x1.35cdb4p-1, 0x1.38ee8ap-1, 0x1.3c07cap-1,
            0x1.3f196ep-1, 0x1.42236ep-1, 0x1.4525c8p-1, 0x1.482074p-1,
            0x1.4b1372p-1, 0x1.4dfebap-1, 0x1.50e24cp-1, 0x1.53be26p-1,
            0x1.569244p-1, 0x1.595ea6p-1, 0x1.5c2348p-1, 0x1.5ee02ep-1,
            0x1.619556p-1, 0x1.6442c0p-1, 0x1.66e86ep-1, 0x1.69865ep-1,
            0x1.6c1c98p-1, 0x1.6eab18p-1, 0x1.7131e6p-1, 0x1.73b102p-1,
            0x1.762870p-1, 0x1.789836p-1, 0x1.7b0058p-1, 0x1.7d60d8p-1,
            0x1.7fb9c0p-1, 0x1.820b12p-1, 0x1.8454d6p-1, 0x1.869712p-1,
            0x1.88d1cep-1, 0x1.8b050ep-1, 0x1.8d30dep-1, 0x1.8f5544p-1,
            0x1.91724ap-1, 0x1.9387f6p-1, 0x1.959652p-1, 0x1.979d68p-1,
            0x1.999d42p-1, 0x1.9b95e8p-1, 0x1.9d8768p-1, 0x1.9f71cap-1,
            0x1.a1551ap-1, 0x1.a33162p-1, 0x1.a506b0p-1, 0x1.a6d50cp-1,
            0x1.a89c86p-1, 0x1.aa5d26p-1, 0x1.ac16fcp-1, 0x1.adca14p-1,
            0x1.af767ap-1, 0x1.b11c3cp-1, 0x1.b2bb68p-1, 0x1.b4540ap-1,
            0x1.b5e630p-1, 0x1.b771e8p-1, 0x1.b8f742p-1, 0x1.ba764ap-1,
            0x1.bbef10p-1, 0x1.bd61a2p-1, 0x1.bece0ep-1, 0x1.c03464p-1,
            0x1.c194b2p-1, 0x1.c2ef08p-1, 0x1.c44376p-1, 0x1.c5920ap-1,
            0x1.c6dad2p-1, 0x1.c81de2p-1, 0x1.c95b46p-1, 0x1.ca930ep-1,
            0x1.cbc54cp-1, 0x1.ccf20cp-1, 0x1.ce1962p-1, 0x1.cf3b5cp-1,
            0x1.d0580cp-1, 0x1.d16f7ep-1, 0x1.d281c4p-1, 0x1.d38ef0p-1,
            0x1.d49710p-1, 0x1.d59a34p-1, 0x1.d6986cp-1, 0x1.d791cap-1,
            0x1.d8865ep-1, 0x1.d97636p-1, 0x1.da6162p-1, 0x1.db47f4p-1,
            0x1.dc29fcp-1, 0x1.dd0788p-1, 0x1.dde0aap-1, 0x1.deb570p-1,
            0x1.df85eap-1, 0x1.e0522ap-1, 0x1.e11a3ep-1, 0x1.e1de36p-1,
            0x1.e29e22p-1, 0x1.e35a12p-1, 0x1.e41214p-1, 0x1.e4c638p-1,
            0x1.e5768cp-1, 0x1.e62322p-1, 0x1.e6cc08p-1, 0x1.e7714ap-1,
            0x1.e812fcp-1, 0x1.e8b12ap-1, 0x1.e94be4p-1, 0x1.e9e336p-1,
            0x1.ea7730p-1, 0x1.eb07e2p-1, 0x1.eb9558p-1, 0x1.ec1fa2p-1,
            0x1.eca6ccp-1, 0x1.ed2ae6p-1, 0x1.edabfcp-1, 0x1.ee2a1ep-1,
            0x1.eea556p-1, 0x1.ef1db4p-1, 0x1.ef9344p-1, 0x1.f00614p-1,
            0x1.f07630p-1, 0x1.f0e3a6p-1, 0x1.f14e82p-1, 0x1.f1b6d0p-1,
            0x1.f21ca0p-1, 0x1.f27ff8p-1, 0x1.f2e0eap-1, 0x1.f33f7ep-1,
            0x1.f39bc2p-1, 0x1.f3f5c2p-1, 0x1.f44d88p-1, 0x1.f4a31ep-1,
            0x1.f4f694p-1, 0x1.f547f2p-1, 0x1.f59742p-1, 0x1.f5e490p-1,
            0x1.f62fe8p-1, 0x1.f67952p-1, 0x1.f6c0dcp-1, 0x1.f7068cp-1,
            0x1.f74a6ep-1, 0x1.f78c8cp-1, 0x1.f7cceep-1, 0x1.f80ba2p-1,
            0x1.f848acp-1, 0x1.f8841ap-1, 0x1.f8bdf2p-1, 0x1.f8f63ep-1,
            0x1.f92d08p-1, 0x1.f96256p-1, 0x1.f99634p-1, 0x1.f9c8a8p-1,
            0x1.f9f9bap-1, 0x1.fa2974p-1, 0x1.fa57dep-1, 0x1.fa84fep-1,
            0x1.fab0dep-1, 0x1.fadb84p-1, 0x1.fb04f6p-1, 0x1.fb2d40p-1,
            0x1.fb5464p-1, 0x1.fb7a6cp-1, 0x1.fb9f60p-1, 0x1.fbc344p-1,
            0x1.fbe61ep-1, 0x1.fc07fap-1, 0x1.fc28d8p-1, 0x1.fc48c2p-1,
            0x1.fc67bcp-1, 0x1.fc85d0p-1, 0x1.fca2fep-1, 0x1.fcbf52p-1,
            0x1.fcdaccp-1, 0x1.fcf576p-1, 0x1.fd0f54p-1, 0x1.fd286ap-1,
            0x1.fd40bep-1, 0x1.fd5856p-1, 0x1.fd6f34p-1, 0x1.fd8562p-1,
            0x1.fd9ae2p-1, 0x1.fdafb8p-1, 0x1.fdc3e8p-1, 0x1.fdd77ap-1,
            0x1.fdea6ep-1, 0x1.fdfcccp-1, 0x1.fe0e96p-1, 0x1.fe1fd0p-1,
            0x1.fe3080p-1, 0x1.fe40a6p-1, 0x1.fe504cp-1, 0x1.fe5f70p-1,
            0x1.fe6e18p-1, 0x1.fe7c46p-1, 0x1.fe8a00p-1, 0x1.fe9748p-1,
            0x1.fea422p-1, 0x1.feb090p-1, 0x1.febc96p-1, 0x1.fec836p-1,
            0x1.fed374p-1, 0x1.fede52p-1, 0x1.fee8d4p-1, 0x1.fef2fep-1,
            0x1.fefccep-1, 0x1.ff064cp-1, 0x1.ff0f76p-1, 0x1.ff1852p-1,
            0x1.ff20e0p-1, 0x1.ff2924p-1, 0x1.ff3120p-1, 0x1.ff38d6p-1,
            0x1.ff4048p-1, 0x1.ff4778p-1, 0x1.ff4e68p-1, 0x1.ff551ap-1,
            0x1.ff5b90p-1, 0x1.ff61ccp-1, 0x1.ff67d0p-1, 0x1.ff6d9ep-1,
            0x1.ff7338p-1, 0x1.ff789ep-1, 0x1.ff7dd4p-1, 0x1.ff82dap-1,
            0x1.ff87b2p-1, 0x1.ff8c5cp-1, 0x1.ff90dcp-1, 0x1.ff9532p-1,
            0x1.ff9960p-1, 0x1.ff9d68p-1, 0x1.ffa14ap-1, 0x1.ffa506p-1,
            0x1.ffa8a0p-1, 0x1.ffac18p-1, 0x1.ffaf6ep-1, 0x1.ffb2a6p-1,
            0x1.ffb5bep-1, 0x1.ffb8b8p-1, 0x1.ffbb98p-1, 0x1.ffbe5ap-1,
            0x1.ffc102p-1, 0x1.ffc390p-1, 0x1.ffc606p-1, 0x1.ffc862p-1,
            0x1.ffcaa8p-1, 0x1.ffccd8p-1, 0x1.ffcef4p-1, 0x1.ffd0fap-1,
            0x1.ffd2eap-1, 0x1.ffd4cap-1, 0x1.ffd696p-1, 0x1.ffd84ep-1,
            0x1.ffd9f8p-1, 0x1.ffdb90p-1, 0x1.ffdd18p-1, 0x1.ffde90p-1,
            0x1.ffdffap-1, 0x1.ffe154p-1, 0x1.ffe2a2p-1, 0x1.ffe3e2p-1,
            0x1.ffe514p-1, 0x1.ffe63cp-1, 0x1.ffe756p-1, 0x1.ffe866p-1,
            0x1.ffe96ap-1, 0x1.ffea64p-1, 0x1.ffeb54p-1, 0x1.ffec3ap-1,
            0x1.ffed16p-1, 0x1.ffedeap-1, 0x1.ffeeb4p-1, 0x1.ffef76p-1,
            0x1.fff032p-1, 0x1.fff0e4p-1, 0x1.fff18ep-1, 0x1.fff232p-1,
            0x1.fff2d0p-1, 0x1.fff366p-1, 0x1.fff3f6p-1, 0x1.fff480p-1,
            0x1.fff504p-1, 0x1.fff582p-1, 0x1.fff5fcp-1, 0x1.fff670p-1,
            0x1.fff6dep-1, 0x1.fff74ap-1, 0x1.fff7aep-1, 0x1.fff810p-1,
            0x1.fff86cp-1, 0x1.fff8c6p-1, 0x1.fff91cp-1, 0x1.fff96cp-1,
            0x1.fff9bap-1, 0x1.fffa04p-1, 0x1.fffa4cp-1, 0x1.fffa90p-1,
            0x1.fffad0p-1, 0x1.fffb0ep-1, 0x1.fffb4ap-1, 0x1.fffb82p-1,
            0x1.fffbb8p-1, 0x1.fffbecp-1, 0x1.fffc1ep-1, 0x1.fffc4ep-1,
            0x1.fffc7ap-1, 0x1.fffca6p-1, 0x1.fffccep-1, 0x1.fffcf6p-1,
            0x1.fffd1ap-1, 0x1.fffd3ep-1, 0x1.fffd60p-1, 0x1.fffd80p-1,
            0x1.fffda0p-1, 0x1.fffdbep-1, 0x1.fffddap-1, 0x1.fffdf4p-1,
            0x1.fffe0ep-1, 0x1.fffe26p-1, 0x1.fffe3ep-1, 0x1.fffe54p-1,
            0x1.fffe68p-1, 0x1.fffe7ep-1, 0x1.fffe90p-1, 0x1.fffea2p-1,
            0x1.fffeb4p-1, 0x1.fffec4p-1, 0x1.fffed4p-1, 0x1.fffee4p-1,
            0x1.fffef2p-1, 0x1.ffff00p-1, 0x1.ffff0cp-1, 0x1.ffff18p-1,
            0x1.ffff24p-1, 0x1.ffff30p-1, 0x1.ffff3ap-1, 0x1.ffff44p-1,
            0x1.ffff4ep-1, 0x1.ffff56p-1, 0x1.ffff60p-1, 0x1.ffff68p-1,
            0x1.ffff70p-1, 0x1.ffff78p-1, 0x1.ffff7ep-1, 0x1.ffff84p-1,
            0x1.ffff8cp-1, 0x1.ffff92p-1, 0x1.ffff98p-1, 0x1.ffff9cp-1,
            0x1.ffffa2p-1, 0x1.ffffa6p-1, 0x1.ffffacp-1, 0x1.ffffb0p-1,
            0x1.ffffb4p-1, 0x1.ffffb8p-1, 0x1.ffffbcp-1, 0x1.ffffc0p-1,
            0x1.ffffc4p-1, 0x1.ffffc6p-1, 0x1.ffffcap-1, 0x1.ffffccp-1,
            0x1.ffffd0p-1, 0x1.ffffd2p-1, 0x1.ffffd4p-1, 0x1.ffffd6p-1,
            0x1.ffffd8p-1, 0x1.ffffdcp-1, 0x1.ffffdep-1, 0x1.ffffdep-1,
            0x1.ffffe0p-1, 0x1.ffffe2p-1, 0x1.ffffe4p-1, 0x1.ffffe6p-1,
            0x1.ffffe8p-1, 0x1.ffffe8p-1, 0x1.ffffeap-1, 0x1.ffffeap-1,
            0x1.ffffecp-1, 0x1.ffffeep-1, 0x1.ffffeep-1, 0x1.fffff0p-1,
            0x1.fffff0p-1, 0x1.fffff2p-1, 0x1.fffff2p-1, 0x1.fffff2p-1,
            0x1.fffff4p-1, 0x1.fffff4p-1, 0x1.fffff4p-1, 0x1.fffff6p-1,
            0x1.fffff6p-1, 0x1.fffff6p-1, 0x1.fffff8p-1, 0x1.fffff8p-1,
            0x1.fffff8p-1, 0x1.fffff8p-1, 0x1.fffffap-1, 0x1.fffffap-1,
            0x1.fffffap-1, 0x1.fffffap-1, 0x1.fffffap-1, 0x1.fffffap-1,
            0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1,
            0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1, 0x1.fffffcp-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1, 0x1.fffffep-1,
            0x1.fffffep-1, 0x1.fffffep-1, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0, 0x1.000000p+0,
            0x1.000000p+0,
        },
    .scale =
        {
            0x1.20dd76p+0,  0x1.20dd76p+0,  0x1.20cb68p+0,  0x1.20b4d8p+0,
            0x1.209546p+0,  0x1.206cb4p+0,  0x1.203b26p+0,  0x1.2000a0p+0,
            0x1.1fbd28p+0,  0x1.1f70c4p+0,  0x1.1f1b7ap+0,  0x1.1ebd56p+0,
            0x1.1e565cp+0,  0x1.1de698p+0,  0x1.1d6e14p+0,  0x1.1cecdcp+0,
            0x1.1c62fap+0,  0x1.1bd07cp+0,  0x1.1b3572p+0,  0x1.1a91e6p+0,
            0x1.19e5eap+0,  0x1.19318cp+0,  0x1.1874dep+0,  0x1.17aff0p+0,
            0x1.16e2d8p+0,  0x1.160da4p+0,  0x1.153068p+0,  0x1.144b3cp+0,
            0x1.135e30p+0,  0x1.12695ep+0,  0x1.116cd8p+0,  0x1.1068bap+0,
            0x1.0f5d16p+0,  0x1.0e4a08p+0,  0x1.0d2fa6p+0,  0x1.0c0e0ap+0,
            0x1.0ae550p+0,  0x1.09b590p+0,  0x1.087ee4p+0,  0x1.07416cp+0,
            0x1.05fd3ep+0,  0x1.04b27cp+0,  0x1.036140p+0,  0x1.0209a6p+0,
            0x1.00abd0p+0,  0x1.fe8fb0p-1,  0x1.fbbbbep-1,  0x1.f8dc0ap-1,
            0x1.f5f0cep-1,  0x1.f2fa4cp-1,  0x1.eff8c4p-1,  0x1.ecec78p-1,
            0x1.e9d5a8p-1,  0x1.e6b498p-1,  0x1.e38988p-1,  0x1.e054bep-1,
            0x1.dd167cp-1,  0x1.d9cf06p-1,  0x1.d67ea2p-1,  0x1.d32592p-1,
            0x1.cfc41ep-1,  0x1.cc5a8ap-1,  0x1.c8e91cp-1,  0x1.c5701ap-1,
            0x1.c1efcap-1,  0x1.be6872p-1,  0x1.bada5ap-1,  0x1.b745c6p-1,
            0x1.b3aafcp-1,  0x1.b00a46p-1,  0x1.ac63e8p-1,  0x1.a8b828p-1,
            0x1.a5074ep-1,  0x1.a1519ep-1,  0x1.9d9762p-1,  0x1.99d8dap-1,
            0x1.961650p-1,  0x1.925008p-1,  0x1.8e8646p-1,  0x1.8ab950p-1,
            0x1.86e96ap-1,  0x1.8316d6p-1,  0x1.7f41dcp-1,  0x1.7b6abcp-1,
            0x1.7791b8p-1,  0x1.73b714p-1,  0x1.6fdb12p-1,  0x1.6bfdf0p-1,
            0x1.681ff2p-1,  0x1.644156p-1,  0x1.60625cp-1,  0x1.5c8342p-1,
            0x1.58a446p-1,  0x1.54c5a6p-1,  0x1.50e79ep-1,  0x1.4d0a68p-1,
            0x1.492e42p-1,  0x1.455366p-1,  0x1.417a0cp-1,  0x1.3da26ep-1,
            0x1.39ccc2p-1,  0x1.35f940p-1,  0x1.32281ep-1,  0x1.2e5992p-1,
            0x1.2a8dcep-1,  0x1.26c508p-1,  0x1.22ff72p-1,  0x1.1f3d3cp-1,
            0x1.1b7e98p-1,  0x1.17c3b6p-1,  0x1.140cc4p-1,  0x1.1059eep-1,
            0x1.0cab62p-1,  0x1.09014cp-1,  0x1.055bd6p-1,  0x1.01bb2cp-1,
            0x1.fc3ee6p-2,  0x1.f511aap-2,  0x1.edeeeep-2,  0x1.e6d700p-2,
            0x1.dfca26p-2,  0x1.d8c8aap-2,  0x1.d1d2d0p-2,  0x1.cae8dap-2,
            0x1.c40b08p-2,  0x1.bd3998p-2,  0x1.b674c8p-2,  0x1.afbcd4p-2,
            0x1.a911f0p-2,  0x1.a27456p-2,  0x1.9be438p-2,  0x1.9561c8p-2,
            0x1.8eed36p-2,  0x1.8886b2p-2,  0x1.822e66p-2,  0x1.7be47ap-2,
            0x1.75a91ap-2,  0x1.6f7c6ap-2,  0x1.695e8cp-2,  0x1.634fa6p-2,
            0x1.5d4fd4p-2,  0x1.575f34p-2,  0x1.517de6p-2,  0x1.4bac00p-2,
            0x1.45e99cp-2,  0x1.4036d0p-2,  0x1.3a93b2p-2,  0x1.350052p-2,
            0x1.2f7cc4p-2,  0x1.2a0916p-2,  0x1.24a554p-2,  0x1.1f518ap-2,
            0x1.1a0dc6p-2,  0x1.14da0ap-2,  0x1.0fb662p-2,  0x1.0aa2d0p-2,
            0x1.059f5ap-2,  0x1.00ac00p-2,  0x1.f79184p-3,  0x1.edeb40p-3,
            0x1.e46530p-3,  0x1.daff4ap-3,  0x1.d1b982p-3,  0x1.c893cep-3,
            0x1.bf8e1cp-3,  0x1.b6a856p-3,  0x1.ade26cp-3,  0x1.a53c42p-3,
            0x1.9cb5bep-3,  0x1.944ec2p-3,  0x1.8c0732p-3,  0x1.83deeap-3,
            0x1.7bd5c8p-3,  0x1.73eba4p-3,  0x1.6c2056p-3,  0x1.6473b6p-3,
            0x1.5ce596p-3,  0x1.5575c8p-3,  0x1.4e241ep-3,  0x1.46f066p-3,
            0x1.3fda6cp-3,  0x1.38e1fap-3,  0x1.3206dcp-3,  0x1.2b48dap-3,
            0x1.24a7b8p-3,  0x1.1e233ep-3,  0x1.17bb2cp-3,  0x1.116f48p-3,
            0x1.0b3f52p-3,  0x1.052b0cp-3,  0x1.fe6460p-4,  0x1.f2a902p-4,
            0x1.e72372p-4,  0x1.dbd32ap-4,  0x1.d0b7a0p-4,  0x1.c5d04ap-4,
            0x1.bb1c98p-4,  0x1.b09bfcp-4,  0x1.a64de6p-4,  0x1.9c31c6p-4,
            0x1.92470ap-4,  0x1.888d1ep-4,  0x1.7f036cp-4,  0x1.75a960p-4,
            0x1.6c7e64p-4,  0x1.6381e2p-4,  0x1.5ab342p-4,  0x1.5211ecp-4,
            0x1.499d48p-4,  0x1.4154bcp-4,  0x1.3937b2p-4,  0x1.31458ep-4,
            0x1.297dbap-4,  0x1.21df9ap-4,  0x1.1a6a96p-4,  0x1.131e14p-4,
            0x1.0bf97ep-4,  0x1.04fc3ap-4,  0x1.fc4b5ep-5,  0x1.eeea8cp-5,
            0x1.e1d4d0p-5,  0x1.d508fap-5,  0x1.c885e0p-5,  0x1.bc4a54p-5,
            0x1.b05530p-5,  0x1.a4a54ap-5,  0x1.99397ap-5,  0x1.8e109cp-5,
            0x1.83298ep-5,  0x1.78832cp-5,  0x1.6e1c58p-5,  0x1.63f3f6p-5,
            0x1.5a08e8p-5,  0x1.505a18p-5,  0x1.46e66cp-5,  0x1.3dacd2p-5,
            0x1.34ac36p-5,  0x1.2be38cp-5,  0x1.2351c2p-5,  0x1.1af5d2p-5,
            0x1.12ceb4p-5,  0x1.0adb60p-5,  0x1.031ad6p-5,  0x1.f7182ap-6,
            0x1.e85c44p-6,  0x1.da0006p-6,  0x1.cc0180p-6,  0x1.be5ecep-6,
            0x1.b1160ap-6,  0x1.a4255ap-6,  0x1.978ae8p-6,  0x1.8b44e6p-6,
            0x1.7f5188p-6,  0x1.73af0cp-6,  0x1.685bb6p-6,  0x1.5d55ccp-6,
            0x1.529b9ep-6,  0x1.482b84p-6,  0x1.3e03d8p-6,  0x1.3422fep-6,
            0x1.2a875cp-6,  0x1.212f62p-6,  0x1.181984p-6,  0x1.0f443ep-6,
            0x1.06ae14p-6,  0x1.fcab14p-7,  0x1.ec7262p-7,  0x1.dcaf36p-7,
            0x1.cd5ecap-7,  0x1.be7e5ap-7,  0x1.b00b38p-7,  0x1.a202bep-7,
            0x1.94624ep-7,  0x1.87275ep-7,  0x1.7a4f6ap-7,  0x1.6dd7fep-7,
            0x1.61beaep-7,  0x1.56011cp-7,  0x1.4a9cf6p-7,  0x1.3f8ff6p-7,
            0x1.34d7dcp-7,  0x1.2a727ap-7,  0x1.205dacp-7,  0x1.169756p-7,
            0x1.0d1d6ap-7,  0x1.03ede2p-7,  0x1.f60d8ap-8,  0x1.e4cc4ap-8,
            0x1.d4143ap-8,  0x1.c3e1a6p-8,  0x1.b430ecp-8,  0x1.a4fe84p-8,
            0x1.9646f4p-8,  0x1.8806d8p-8,  0x1.7a3adep-8,  0x1.6cdfccp-8,
            0x1.5ff276p-8,  0x1.536fc2p-8,  0x1.4754acp-8,  0x1.3b9e40p-8,
            0x1.30499cp-8,  0x1.2553eep-8,  0x1.1aba78p-8,  0x1.107a8cp-8,
            0x1.06918cp-8,  0x1.f9f9d0p-9,  0x1.e77448p-9,  0x1.d58da6p-9,
            0x1.c4412cp-9,  0x1.b38a3ap-9,  0x1.a36454p-9,  0x1.93cb12p-9,
            0x1.84ba30p-9,  0x1.762d84p-9,  0x1.682100p-9,  0x1.5a90b0p-9,
            0x1.4d78bcp-9,  0x1.40d564p-9,  0x1.34a306p-9,  0x1.28de12p-9,
            0x1.1d8318p-9,  0x1.128ebap-9,  0x1.07fdb4p-9,  0x1.fb99b8p-10,
            0x1.e7f232p-10, 0x1.d4fed8p-10, 0x1.c2b9d0p-10, 0x1.b11d70p-10,
            0x1.a02436p-10, 0x1.8fc8c8p-10, 0x1.8005f0p-10, 0x1.70d6a4p-10,
            0x1.6235fcp-10, 0x1.541f34p-10, 0x1.468daep-10, 0x1.397ceep-10,
            0x1.2ce898p-10, 0x1.20cc76p-10, 0x1.15246ep-10, 0x1.09ec86p-10,
            0x1.fe41cep-11, 0x1.e97ba4p-11, 0x1.d57f52p-11, 0x1.c245d4p-11,
            0x1.afc85ep-11, 0x1.9e0058p-11, 0x1.8ce75ep-11, 0x1.7c7744p-11,
            0x1.6caa0ep-11, 0x1.5d79ecp-11, 0x1.4ee142p-11, 0x1.40daa4p-11,
            0x1.3360ccp-11, 0x1.266ea8p-11, 0x1.19ff46p-11, 0x1.0e0de8p-11,
            0x1.0295f0p-11, 0x1.ef25d4p-12, 0x1.da0110p-12, 0x1.c5b542p-12,
            0x1.b23a5ap-12, 0x1.9f8894p-12, 0x1.8d986ap-12, 0x1.7c629ap-12,
            0x1.6be022p-12, 0x1.5c0a38p-12, 0x1.4cda54p-12, 0x1.3e4a24p-12,
            0x1.305390p-12, 0x1.22f0b4p-12, 0x1.161be4p-12, 0x1.09cfa4p-12,
            0x1.fc0d56p-13, 0x1.e577bcp-13, 0x1.cfd4a6p-13, 0x1.bb1a96p-13,
            0x1.a74068p-13, 0x1.943d4ap-13, 0x1.8208bcp-13, 0x1.709a8ep-13,
            0x1.5feadap-13, 0x1.4ff208p-13, 0x1.40a8c2p-13, 0x1.3207fcp-13,
            0x1.2408eap-13, 0x1.16a502p-13, 0x1.09d5f8p-13, 0x1.fb2b7ap-14,
            0x1.e3bcf4p-14, 0x1.cd5528p-14, 0x1.b7e946p-14, 0x1.a36eecp-14,
            0x1.8fdc1cp-14, 0x1.7d2738p-14, 0x1.6b4702p-14, 0x1.5a329cp-14,
            0x1.49e178p-14, 0x1.3a4b60p-14, 0x1.2b6876p-14, 0x1.1d3120p-14,
            0x1.0f9e1cp-14, 0x1.02a868p-14, 0x1.ec929ap-15, 0x1.d4f4b4p-15,
            0x1.be6abcp-15, 0x1.a8e8ccp-15, 0x1.94637ep-15, 0x1.80cfdcp-15,
            0x1.6e2368p-15, 0x1.5c540cp-15, 0x1.4b581cp-15, 0x1.3b2652p-15,
            0x1.2bb5ccp-15, 0x1.1cfe02p-15, 0x1.0ef6c4p-15, 0x1.019842p-15,
            0x1.e9b5e8p-16, 0x1.d16f58p-16, 0x1.ba4f04p-16, 0x1.a447b8p-16,
            0x1.8f4cccp-16, 0x1.7b5224p-16, 0x1.684c22p-16, 0x1.562facp-16,
            0x1.44f21ep-16, 0x1.34894ap-16, 0x1.24eb72p-16, 0x1.160f44p-16,
            0x1.07ebd2p-16, 0x1.f4f12ep-17, 0x1.db5ad0p-17, 0x1.c304f0p-17,
            0x1.abe09ep-17, 0x1.95df98p-17, 0x1.80f43ap-17, 0x1.6d1178p-17,
            0x1.5a2ae0p-17, 0x1.483488p-17, 0x1.372310p-17, 0x1.26eb9ep-17,
            0x1.1783cep-17, 0x1.08e1bap-17, 0x1.f5f7d8p-18, 0x1.db92b6p-18,
            0x1.c282cep-18, 0x1.aab7acp-18, 0x1.94219cp-18, 0x1.7eb1a2p-18,
            0x1.6a5972p-18, 0x1.570b6ap-18, 0x1.44ba86p-18, 0x1.335a62p-18,
            0x1.22df2ap-18, 0x1.133d96p-18, 0x1.046aeap-18, 0x1.ecb9d0p-19,
            0x1.d21398p-19, 0x1.b8d094p-19, 0x1.a0df10p-19, 0x1.8a2e26p-19,
            0x1.74adc8p-19, 0x1.604ea8p-19, 0x1.4d0232p-19, 0x1.3aba86p-19,
            0x1.296a70p-19, 0x1.190562p-19, 0x1.097f62p-19, 0x1.f59a20p-20,
            0x1.d9c736p-20, 0x1.bf716cp-20, 0x1.a6852cp-20, 0x1.8eefd8p-20,
            0x1.789fb8p-20, 0x1.6383f8p-20, 0x1.4f8c96p-20, 0x1.3caa62p-20,
            0x1.2acee2p-20, 0x1.19ec60p-20, 0x1.09f5d0p-20, 0x1.f5bd96p-21,
            0x1.d9371ep-21, 0x1.be41dep-21, 0x1.a4c89ep-21, 0x1.8cb738p-21,
            0x1.75fa8ep-21, 0x1.608078p-21, 0x1.4c37c0p-21, 0x1.39100ep-21,
            0x1.26f9e0p-21, 0x1.15e682p-21, 0x1.05c804p-21, 0x1.ed2254p-22,
            0x1.d06ad6p-22, 0x1.b551c8p-22, 0x1.9bc0a0p-22, 0x1.83a200p-22,
            0x1.6ce1aap-22, 0x1.576c72p-22, 0x1.43302cp-22, 0x1.301ba2p-22,
            0x1.1e1e86p-22, 0x1.0d2966p-22, 0x1.fa5b50p-23, 0x1.dc3ae4p-23,
            0x1.bfd756p-23, 0x1.a517dap-23, 0x1.8be4f8p-23, 0x1.74287ep-23,
            0x1.5dcd66p-23, 0x1.48bfd4p-23, 0x1.34ecf8p-23, 0x1.224310p-23,
            0x1.10b148p-23,
        },
};

static inline __m256 erf_ps(__m256 x) {
    __m256i zero = _mm256_setzero_si256();
    __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    /* |x| > 1/64 - 1/512. */
    __m256 gt_min_mask =
        _mm256_cmp_ps(a, _mm256_set1_ps(0x1.cp-7f), _CMP_GT_OS);
    int gt_min_mask_as_int;
    std::memcpy(&gt_min_mask_as_int, &gt_min_mask, 1);
    __m256 tmp_i = _mm256_mul_ps(a, _mm256_set1_ps(128.f));

    __m256i signed_i = _mm256_cvttps_epi32(
        _mm256_round_ps(tmp_i, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);
    __m256i i = _mm256_and_si256(signed_i, mask);
    i = _mm256_blendv_epi8(zero, i, _mm256_castps_si256(gt_min_mask));
    i = _mm256_min_epu32(i, _mm256_set1_epi32(512));
    __m256 tmp_r = _mm256_cvtepi32_ps(i);
    i = _mm256_mullo_epi32(i, _mm256_set1_epi32(1));
    __m256 r = _mm256_mul_ps(tmp_r, _mm256_set1_ps(1.f / 128));
    __m256 erfr = _mm256_i32gather_ps(__sv_erff_data.erf, i, sizeof(float));
    __m256 scale = _mm256_i32gather_ps(__sv_erff_data.scale, i, sizeof(float));
    __m256 ge_max_mask = _mm256_cmp_ps(a, _mm256_set1_ps(3.9375f), _CMP_GE_OS);
    std::memcpy(&gt_min_mask_as_int, &ge_max_mask, 1);
    /* erf(x) ~ erf(r) + scale * d * (1 - r * d - 1/3 * d^2). */
    __m256 d = _mm256_sub_ps(a, r);
    __m256 d2 = _mm256_mul_ps(d, d);
    __m256 y = _mm256_fmadd_ps(d, _mm256_set1_ps(0x1.555556p-2f), r);
    y = _mm256_fnmadd_ps(y, d2, d);
    y = _mm256_fmadd_ps(y, scale, erfr);
    y = _mm256_blendv_ps(y, _mm256_set1_ps(1.f), ge_max_mask);

    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    __m256 x_sign = _mm256_and_ps(x, sign_mask);
    __m256 y_abs = _mm256_andnot_ps(sign_mask, y);
    y = _mm256_or_ps(y_abs, x_sign);

    return y;
}

static inline __m256 atan2256_ps(__m256 y, __m256 x) {
    // Reference: https://mazzo.li/posts/vectorized-atan2.html

    const __m256 magic_zero = _mm256_set1_ps(0.0f);
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_pi = _mm256_set1_ps(3.1415927f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);

    // not_equal_zero_x = (x != 0.0f);
    __m256 not_equal_zero_x = _mm256_cmp_ps(x, magic_zero, _CMP_NEQ_OQ);

    // not_equal_zero_y = (y != 0.0f);
    __m256 not_equal_zero_y = _mm256_cmp_ps(y, magic_zero, _CMP_NEQ_OQ);

    // normal_mode = ((x != 0.0f) & (y != 0.0f));
    __m256 normal_mode = _mm256_and_ps(not_equal_zero_x, not_equal_zero_y);

    // negative_mask_x = magic_negative_zero && x;
    __m256 negative_mask_x = _mm256_and_ps(magic_negative_zero, x);

    // negative_mask_y = magic_negative_zero && y;
    __m256 negative_mask_y = _mm256_and_ps(magic_negative_zero, y);

    // pi_additions = ((x < 0.0f) ? ((y < 0.0f) ? -PI : PI) : 0.0f);
    __m256 pi_additions = _mm256_and_ps(
        _mm256_cmp_ps(x, magic_zero, _CMP_LT_OQ),
        _mm256_or_ps(_mm256_and_ps(_mm256_cmp_ps(y, magic_zero, _CMP_LT_OQ),
                                   magic_negative_zero),
                     magic_pi));

    // normal_result = (atan(y / x) + pi_additions);
    __m256 normal_result =
        _mm256_add_ps(atan256_ps(_mm256_div_ps(y, x)), pi_additions);

    // negative_mask_full_x = ((negative_mask_x | PI) < 0.0f);
    __m256 negative_mask_full_x = _mm256_cmp_ps(
        _mm256_or_ps(negative_mask_x, magic_pi), magic_zero, _CMP_LT_OQ);

    // x1 = (negative_mask_y ? -(0.5 * PI) : (0.5 * PI));
    // x2 = (negative_mask_full_x ? PI : 0.0f);
    // special_result = ((y != 0.0f) ? x1 : x2);
    __m256 special_result = _mm256_or_ps(
        _mm256_and_ps(not_equal_zero_y,
                      _mm256_or_ps(negative_mask_y, magic_half_pi)),
        _mm256_andnot_ps(
            not_equal_zero_y,
            _mm256_or_ps(_mm256_and_ps(negative_mask_full_x, magic_pi),
                         _mm256_andnot_ps(negative_mask_full_x, magic_zero))));

    // return (normal_mode ? normal_result : special_result);
    return _mm256_or_ps(_mm256_and_ps(normal_mode, normal_result),
                        _mm256_andnot_ps(normal_mode, special_result));
}

static inline __m256 abs256_ps(__m256 x) {
    // Use negative zero as the sign bit mask.
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(magic_negative_zero, x);
}

#endif // AVX_MATHFUN_H