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
#include "../../native_vector.h"
#include <immintrin.h>

#ifndef NTT_VLEN
constexpr size_t NTT_VLEN = sizeof(__m256i) * 8;
#endif

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(int8_t, __m256i, __v32qs, 32)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(uint8_t, __m256i, __v32qu, 32)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(int16_t, __m256i, __v16hi, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(uint16_t, __m256i, __v16hu, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(int32_t, __m256i, __v8si, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(uint32_t, __m256i, __v8su, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int64_t, __m256i, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint64_t, __m256i, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(float, __m256, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(double, __m256d, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

// mask vectors
#define NTT_DEFINE_NATIVE_MASK_VECTOR(bits, cast_type)                         \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(bool, __m256i, NTT_VLEN / bits)             \
                                                                               \
    template <Dimensions TIndex>                                               \
    static bool get_element(const __m256i &array,                              \
                            const TIndex &index) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const auto casted_value = (cast_type)array;                            \
        const auto offset = (size_t)index[dim_zero];                           \
        return casted_value[offset] != 0;                                      \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(__m256i &array, const TIndex &index,               \
                            bool value) noexcept {                             \
        using casted_element_type = std::decay_t<decltype(cast_type{}[0])>;    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        auto &casted_value = reinterpret_cast<cast_type &>(array);             \
        const auto offset = (size_t)index[dim_zero];                           \
        casted_value[offset] =                                                 \
            value ? (casted_element_type(1) << (bits - 1)) : 0;                \
    }                                                                          \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_MASK_VECTOR(8, __v32qi)
NTT_DEFINE_NATIVE_MASK_VECTOR(16, __v16hi)
NTT_DEFINE_NATIVE_MASK_VECTOR(32, __v8si)
NTT_DEFINE_NATIVE_MASK_VECTOR(64, __v4di)

#undef NTT_DEFINE_NATIVE_MASK_VECTOR
