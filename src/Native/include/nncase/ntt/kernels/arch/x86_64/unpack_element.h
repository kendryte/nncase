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
#include <immintrin.h>
#include <array>

inline void unpack_elemt(std::array<float, 4> &arr, const __m128 &vec) {
    _mm_store_ps(&arr[0], vec);
}

inline void unpack_elemt(std::array<float, 8> &arr, const __m256 &vec) {
    _mm256_store_ps(&arr[0], vec);
}