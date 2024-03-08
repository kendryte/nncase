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
#include "../../../vector_type.h"
#include "avx_mathfun.h"
#include <immintrin.h>

namespace std {
inline __m256 cos(__m256 v) {
    __m256 s, c;
    sincos256_ps(v, &s, &c);
    return s;
}

inline __m128 cos(__m128 v) {
    float arr[4];
    _mm_store_ps(arr, v);
    for (size_t i = 0; i < 4; i++) {
        arr[i] = cosf(arr[i]);
    }
    return _mm_load_ps(arr);
}

inline __m128 sqrt(__m128 v) { return _mm_sqrt_ps(v); }
} // namespace std

namespace nncase::ntt::arch {
template <size_t Extent, class T, class Op>
constexpr void unary(Op &&op, const T *input_p, T *output_p) {
    for (size_t i = 0; i < Extent; i++) {
        output_p[i] = op(input_p[i]);
    }
}

template <class T, class Op>
constexpr void unary(Op &&op, const T *input_p, T *output_p, size_t extent) {
    for (size_t i = 0; i < extent; i++) {
        output_p[i] = op(input_p[i]);
    }
}
} // namespace nncase::ntt::arch

// namespace nncase::ntt::mathops {
// template <> struct sqrt<ntt::vector<float, 4>> {
//     ntt::vector<float, 4> operator()(ntt::vector<float, 4> v) const noexcept
//     {
//         return std::sqrt(v);
//     }
// };

// } // namespace nncase::ntt::mathops