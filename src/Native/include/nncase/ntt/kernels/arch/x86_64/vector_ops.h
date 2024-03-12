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

namespace nncase::ntt::vector_ops {
template <> struct reduce_sum<ntt::vector<float, 8>> {
    float operator()(ntt::vector<float, 8> v) const noexcept {
        // horizontal add top lane and bottom lane
        auto res0 = _mm256_hadd_ps(v, v);
        res0 = _mm256_hadd_ps(res0, res0);
        __m128 acc1 = _mm256_extractf128_ps(res0, 0);
        __m128 acc2 = _mm256_extractf128_ps(res0, 1);
        acc1 = _mm_add_ss(acc1, acc2);
        return _mm_cvtss_f32(acc1);
    }
};

template <> struct reduce_max<ntt::vector<float, 8>> {
    float operator()(ntt::vector<float, 8> v) const noexcept {
        __m128 lhs = _mm256_extractf128_ps(v, 0);
        __m128 rhs = _mm256_extractf128_ps(v, 1);
        __m128 r = _mm_max_ps(lhs, rhs); // a,b,c,d

        __m128 h = _mm_unpackhi_ps(r, r); // c,d,c,d
        __m128 l = _mm_unpacklo_ps(r, r); // a,b,a,b
        r = _mm_max_ps(l, h);             // max(a,c),max(b,d), ...
        return std::max(r[0], r[1]);
    }
};

} // namespace nncase::ntt::vector_ops
