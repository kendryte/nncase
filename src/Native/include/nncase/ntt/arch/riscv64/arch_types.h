/* Copyright 2019-2024 Canaan Inc.
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
#include "../../native_vector.h"

#ifdef __riscv_vector
#include <riscv_vector.h>

#ifndef __riscv_v_fixed_vlen
#error "-mrvv-vector-bits=zvl must be specified in toolchain compiler option."
#endif

#ifndef NTT_VLEN
#define NTT_VLEN __riscv_v_fixed_vlen
#endif

#ifndef NTT_VL
#define NTT_VL(sew, lmul) ((NTT_VLEN) / (sew) * (lmul))
#endif

// rvv fixed type
#define REGISTER_RVV_FIXED_TYPE_WITH_LMUL(lmul)                                \
    typedef vint8m##lmul##_t fixed_vint8m##lmul##_t                            \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vuint8m##lmul##_t fixed_vuint8m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vint16m##lmul##_t fixed_vint16m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vuint16m##lmul##_t fixed_vuint16m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vint32m##lmul##_t fixed_vint32m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vuint32m##lmul##_t fixed_vuint32m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vint64m##lmul##_t fixed_vint64m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vuint64m##lmul##_t fixed_vuint64m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vfloat32m##lmul##_t fixed_vfloat32m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));               \
    typedef vfloat64m##lmul##_t fixed_vfloat64m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(lmul * NTT_VLEN)));

REGISTER_RVV_FIXED_TYPE_WITH_LMUL(1)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL(2)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL(4)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL(8)

// rvv native vector
#define NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL(lmul)                               \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        int8_t, fixed_vint8m##lmul##_t, lmul *NTT_VLEN / 8 / sizeof(int8_t))   \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8m##lmul##_t,   \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(uint8_t))                \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16m##lmul##_t,   \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(int16_t))                \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint16_t, fixed_vuint16m##lmul##_t, \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(uint16_t))               \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, fixed_vint32m##lmul##_t,   \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(int32_t))                \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint32_t, fixed_vuint32m##lmul##_t, \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(uint32_t))               \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int64_t, fixed_vint64m##lmul##_t,   \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(int64_t))                \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint64_t, fixed_vuint64m##lmul##_t, \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(uint64_t))               \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(float, fixed_vfloat32m##lmul##_t,   \
                                           lmul *NTT_VLEN / 8 / sizeof(float)) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(double, fixed_vfloat64m##lmul##_t,  \
                                           lmul *NTT_VLEN / 8 /                \
                                               sizeof(double))                 \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL(1)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL(2)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL(4)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL(8)
#endif