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
#include <arm_neon.h>

NTT_DEFINE_NATIVE_TENSOR(int8_t, int8x16_t, 16)
NTT_DEFINE_NATIVE_TENSOR(uint8_t, uint8x16_t, 16)
NTT_DEFINE_NATIVE_TENSOR(int16_t, int16x8_t, 8)
NTT_DEFINE_NATIVE_TENSOR(uint16_t, uint16x8_t, 8)
NTT_DEFINE_NATIVE_TENSOR(int32_t, int32x4_t, 4)
NTT_DEFINE_NATIVE_TENSOR(uint32_t, uint32x4_t, 4)
NTT_DEFINE_NATIVE_TENSOR(int64_t, int64x2_t, 2)
NTT_DEFINE_NATIVE_TENSOR(uint64_t, uint64x2_t, 2)
NTT_DEFINE_NATIVE_TENSOR(float, float32x4_t, 4)
NTT_DEFINE_NATIVE_TENSOR(float, float32x4x2_t, 8)
NTT_DEFINE_NATIVE_TENSOR(double, float64x2_t, 2)
