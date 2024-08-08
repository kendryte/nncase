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
#include <arm_neon.h>

#ifndef NTT_VLEN
#define NTT_VLEN (sizeof(int8x16_t) * 8)
#endif

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int8_t, int8x16_t, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint8_t, uint8x16_t, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int16_t, int16x8_t, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint16_t, uint16x8_t, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int32_t, int32x4_t, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint32_t, uint32x4_t, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(int64_t, int64x2_t, 2)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(uint64_t, uint64x2_t, 2)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(float, float32x4_t, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(float, float32x4x2_t, 8)
static float get_element(const float32x4x2_t &array,
                         ranked_shape<1> index) noexcept {
    return array.val[index[0] / 4][index[0] % 4];
}

static void set_element(float32x4x2_t &array, ranked_shape<1> index,
                        float value) noexcept {
    array.val[index[0] / 4][index[0] % 4] = value;
}
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(double, float64x2_t, 2)
NTT_END_DEFINE_NATIVE_VECTOR()
