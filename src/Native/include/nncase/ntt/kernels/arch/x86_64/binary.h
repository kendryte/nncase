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
#include <nncase/ntt/vector_type.h>

namespace nncase::ntt::mathops {

template <> struct add<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v1,
                                     ntt::vector<float, 8> v2) const noexcept {
        return _mm256_add_ps(v1, v2);
    }
};
template <> struct sub<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v1,
                                     ntt::vector<float, 8> v2) const noexcept {
        return _mm256_sub_ps(v1, v2);
    }
};
template <> struct mul<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v1,
                                     ntt::vector<float, 8> v2) const noexcept {
        return _mm256_mul_ps(v1, v2);
    }
};
template <> struct div<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v1,
                                     ntt::vector<float, 8> v2) const noexcept {
        return _mm256_div_ps(v1, v2);
    }
};
template <> struct max<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(ntt::vector<float, 8> v1,
                                     ntt::vector<float, 8> v2) const noexcept {
        return _mm256_max_ps(v1, v2);
    }
};
} // namespace nncase::ntt::mathops