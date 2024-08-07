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

NTT_BEGIN_DEFINE_NATIVE_VECTOR(int8_t, __m256i, 32)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(uint8_t, __m256i, 32)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(int16_t, __m256i, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(uint16_t, __m256i, 16)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(int32_t, __m256i, 8)
static int32_t get_element(const __m256i &array,
                           ranked_shape<1> index) noexcept {
    auto lane_idx = index[0] / 4;
    auto lane_off = index[0] % 4;
    auto lane = !lane_idx ? _mm256_extractf128_ps(array, 0)
                          : _mm256_extractf128_ps(array, 1);
    alignas(__m128i) int32_t lane_vals[4];
    _mm_store_si128((__m128i *)lane_vals, lane);
    return lane_vals[lane_off];
}

static void set_element(__m256i &array, ranked_shape<1> index,
                        int32_t value) noexcept {
    auto lane_idx = index[0] / 4;
    auto lane_off = index[0] % 4;
    auto lane = !lane_idx ? _mm256_extractf128_si256(array, 0)
                          : _mm256_extractf128_si256(array, 1);
    alignas(__m128i) int32_t lane_vals[4];
    _mm_store_si128((__m128i *)lane_vals, lane);
    lane_vals[lane_off] = value;
    lane = _mm_load_si128((__m128i *)lane_vals);
    array = !lane_idx ? _mm256_insertf128_si256(array, lane, 0)
                      : _mm256_insertf128_si256(array, lane, 1);
}
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(uint32_t, __m256i, 8)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(int64_t, __m256i, 4)
static int64_t get_element(const __m256i &array,
                           ranked_shape<1> index) noexcept {
    auto lane_idx = index[0] / 2;
    auto lane_off = index[0] % 2;
    auto lane = !lane_idx ? _mm256_extractf128_ps(array, 0)
                          : _mm256_extractf128_ps(array, 1);
    alignas(__m128i) int64_t lane_vals[2];
    _mm_store_si128((__m128i *)lane_vals, lane);
    return lane_vals[lane_off];
}

static void set_element(__m256i &array, ranked_shape<1> index,
                        int64_t value) noexcept {
    auto lane_idx = index[0] / 2;
    auto lane_off = index[0] % 2;
    auto lane = !lane_idx ? _mm256_extractf128_si256(array, 0)
                          : _mm256_extractf128_si256(array, 1);
    alignas(__m128i) int64_t lane_vals[2];
    _mm_store_si128((__m128i *)lane_vals, lane);
    lane_vals[lane_off] = value;
    lane = _mm_load_si128((__m128i *)lane_vals);
    array = !lane_idx ? _mm256_insertf128_si256(array, lane, 0)
                      : _mm256_insertf128_si256(array, lane, 1);
}
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(uint64_t, __m256i, 4)
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(float, __m256, 8)
static float get_element(const __m256 &array, ranked_shape<1> index) noexcept {
    auto lane_idx = index[0] / 4;
    auto lane_off = index[0] % 4;
    auto lane = !lane_idx ? _mm256_extractf128_ps(array, 0)
                          : _mm256_extractf128_ps(array, 1);
    alignas(__m128) float lane_vals[4];
    _mm_store_ps(lane_vals, lane);
    return lane_vals[lane_off];
}

static void set_element(__m256 &array, ranked_shape<1> index,
                        float value) noexcept {
    auto lane_idx = index[0] / 4;
    auto lane_off = index[0] % 4;
    auto lane = !lane_idx ? _mm256_extractf128_ps(array, 0)
                          : _mm256_extractf128_ps(array, 1);
    alignas(__m128) float lane_vals[4];
    _mm_store_ps(lane_vals, lane);
    lane_vals[lane_off] = value;
    lane = _mm_load_ps(lane_vals);
    array = !lane_idx ? _mm256_insertf128_ps(array, lane, 0)
                      : _mm256_insertf128_ps(array, lane, 1);
}
NTT_END_DEFINE_NATIVE_VECTOR()

NTT_BEGIN_DEFINE_NATIVE_VECTOR(double, __m256d, 4)
static double get_element(const __m256d &array,
                          ranked_shape<1> index) noexcept {
    auto lane_idx = index[0] / 2;
    auto lane_off = index[0] % 2;
    auto lane = !lane_idx ? _mm256_extractf128_pd(array, 0)
                          : _mm256_extractf128_pd(array, 1);
    alignas(__m128d) double lane_vals[2];
    _mm_store_pd(lane_vals, lane);
    return lane_vals[lane_off];
}

static void set_element(__m256d &array, ranked_shape<1> index,
                        double value) noexcept {
    auto lane_idx = index[0] / 2;
    auto lane_off = index[0] % 2;
    auto lane = !lane_idx ? _mm256_extractf128_pd(array, 0)
                          : _mm256_extractf128_pd(array, 1);
    alignas(__m128d) double lane_vals[2];
    _mm_store_pd(lane_vals, lane);
    lane_vals[lane_off] = value;
    lane = _mm_load_pd(lane_vals);
    array = !lane_idx ? _mm256_insertf128_pd(array, lane, 0)
                      : _mm256_insertf128_pd(array, lane, 1);
}
NTT_END_DEFINE_NATIVE_VECTOR()
