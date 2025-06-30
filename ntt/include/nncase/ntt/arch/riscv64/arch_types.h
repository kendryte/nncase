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

#ifndef NTT_VL_
#define NTT_VL(sew, op, lmul) ((NTT_VLEN) / (sew)op(lmul))
#endif

// rvv fixed type
#define REGISTER_RVV_FIXED_TYPE_WITH_LMUL_LT1                                  \
    typedef vint8mf2_t fixed_vint8mf2_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vint8mf4_t fixed_vint8mf4_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vint8mf8_t fixed_vint8mf8_t                                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 8)));                  \
    typedef vuint8mf2_t fixed_vuint8mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint8mf4_t fixed_vuint8mf4_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vuint8mf8_t fixed_vuint8mf8_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 8)));                  \
    typedef vint16mf2_t fixed_vint16mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vint16mf4_t fixed_vint16mf4_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vuint16mf2_t fixed_vuint16mf2_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint16mf4_t fixed_vuint16mf4_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 4)));                  \
    typedef vint32mf2_t fixed_vint32mf2_t                                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vuint32mf2_t fixed_vuint32mf2_t                                    \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));                  \
    typedef vfloat32mf2_t fixed_vfloat32mf2_t                                  \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / 2)));

#define REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(lmul)                            \
    typedef vint8m##lmul##_t fixed_vint8m##lmul##_t                            \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint8m##lmul##_t fixed_vuint8m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint16m##lmul##_t fixed_vint16m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint16m##lmul##_t fixed_vuint16m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint32m##lmul##_t fixed_vint32m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint32m##lmul##_t fixed_vuint32m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vint64m##lmul##_t fixed_vint64m##lmul##_t                          \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vuint64m##lmul##_t fixed_vuint64m##lmul##_t                        \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vfloat32m##lmul##_t fixed_vfloat32m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));               \
    typedef vfloat64m##lmul##_t fixed_vfloat64m##lmul##_t                      \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN * lmul)));

REGISTER_RVV_FIXED_TYPE_WITH_LMUL_LT1
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(1)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(2)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(4)
REGISTER_RVV_FIXED_TYPE_WITH_LMUL_GE1(8)

// rvv native vector
#define NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_LT1                                 \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf2_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 2)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf4_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 4)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, fixed_vint8mf8_t,           \
                                           NTT_VLEN / 8 / sizeof(int8_t) / 8)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf4_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 4) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8mf8_t,         \
                                           NTT_VLEN / 8 / sizeof(uint8_t) / 8) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(int16_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16mf4_t,         \
                                           NTT_VLEN / 8 / sizeof(int16_t) / 4) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint16_t, fixed_vuint16mf2_t, NTT_VLEN / 8 / sizeof(uint16_t) / 2)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint16_t, fixed_vuint16mf4_t, NTT_VLEN / 8 / sizeof(uint16_t) / 4)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, fixed_vint32mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(int32_t) / 2) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        uint32_t, fixed_vuint32mf2_t, NTT_VLEN / 8 / sizeof(uint32_t) / 2)     \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(float, fixed_vfloat32mf2_t,         \
                                           NTT_VLEN / 8 / sizeof(float) / 2)   \
    NTT_END_DEFINE_NATIVE_VECTOR()

#define NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(lmul)                           \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        int8_t, fixed_vint8m##lmul##_t, NTT_VLEN / 8 / sizeof(int8_t) * lmul)  \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, fixed_vuint8m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(uint8_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, fixed_vint16m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int16_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint16_t, fixed_vuint16m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint16_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, fixed_vint32m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int32_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint32_t, fixed_vuint32m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint32_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int64_t, fixed_vint64m##lmul##_t,   \
                                           NTT_VLEN / 8 / sizeof(int64_t) *    \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint64_t, fixed_vuint64m##lmul##_t, \
                                           NTT_VLEN / 8 / sizeof(uint64_t) *   \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(                                    \
        float, fixed_vfloat32m##lmul##_t, NTT_VLEN / 8 / sizeof(float) * lmul) \
    NTT_END_DEFINE_NATIVE_VECTOR()                                             \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(double, fixed_vfloat64m##lmul##_t,  \
                                           NTT_VLEN / 8 / sizeof(double) *     \
                                               lmul)                           \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_LT1
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(1)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(2)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(4)
NTT_DEFINE_NATIVE_VECTOR_WITH_LMUL_GE1(8)

// mask vectors
#define NTT_DEFINE_NATIVE_MASK_VECTOR(bits)                                    \
    typedef vbool##bits##_t fixed_vbool##bits##_t                              \
        __attribute__((riscv_rvv_vector_bits(NTT_VLEN / bits)));               \
                                                                               \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(bool, fixed_vbool##bits##_t,                \
                                   NTT_VLEN / bits)                            \
                                                                               \
    template <Dimensions TIndex>                                               \
    static bool get_element(const fixed_vbool##bits##_t &array,                \
                            const TIndex &index) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const fixed_vint8m1_t i8_value =                                       \
            __riscv_vreinterpret_v_b##bits##_i8m1(array);                      \
        const auto offset = (size_t)index[dim_zero];                           \
        return (i8_value[offset / 8] & (1 << (offset % 8))) != 0;              \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(fixed_vbool##bits##_t &array, const TIndex &index, \
                            bool value) noexcept {                             \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        fixed_vint8m1_t i8_value =                                             \
            __riscv_vreinterpret_v_b##bits##_i8m1(array);                      \
        const auto offset = (size_t)index[dim_zero];                           \
        const auto mask = ~(1 << (offset % 8));                                \
        i8_value[offset / 8] =                                                 \
            (i8_value[offset / 8] & mask) | ((value ? 1 : 0) << (offset % 8)); \
        array = __riscv_vreinterpret_v_i8m1_b##bits(i8_value);                 \
    }                                                                          \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_MASK_VECTOR(1)
NTT_DEFINE_NATIVE_MASK_VECTOR(2)
NTT_DEFINE_NATIVE_MASK_VECTOR(4)
NTT_DEFINE_NATIVE_MASK_VECTOR(8)
NTT_DEFINE_NATIVE_MASK_VECTOR(16)
NTT_DEFINE_NATIVE_MASK_VECTOR(32)
NTT_DEFINE_NATIVE_MASK_VECTOR(64)

#undef NTT_DEFINE_NATIVE_MASK_VECTOR
#endif
