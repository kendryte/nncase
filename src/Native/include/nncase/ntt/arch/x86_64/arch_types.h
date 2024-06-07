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
#include "../../native_tensor.h"
#include <immintrin.h>

#ifndef NTT_VLEN
#define NTT_VLEN (sizeof(__m256i) * 8)
#endif

NTT_DEFINE_NATIVE_TENSOR(int8_t, __m256i, 32)
NTT_DEFINE_NATIVE_TENSOR(uint8_t, __m256i, 32)
NTT_DEFINE_NATIVE_TENSOR(int16_t, __m256i, 16)
NTT_DEFINE_NATIVE_TENSOR(uint16_t, __m256i, 16)
NTT_DEFINE_NATIVE_TENSOR(int32_t, __m256i, 8)
NTT_DEFINE_NATIVE_TENSOR(uint32_t, __m256i, 8)
NTT_DEFINE_NATIVE_TENSOR(int64_t, __m256i, 4)
NTT_DEFINE_NATIVE_TENSOR(uint64_t, __m256i, 4)
NTT_DEFINE_NATIVE_TENSOR(float, __m256, 8)
NTT_DEFINE_NATIVE_TENSOR(double, __m256d, 4)
