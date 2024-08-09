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
#define NTT_VLEN (sizeof(__m256i) * 8)
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
