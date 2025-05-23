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
#include "../../vector_ops.h"
#include "arch_types.h"
#include "avx_mathfun.h"

namespace nncase::ntt::vector_ops {
template <> struct vload_scalar<ntt::vector<float, 8>> {
    ntt::vector<float, 8> operator()(float v) const noexcept {
        return _mm256_set1_ps(v);
    }
};
template <> struct vload_scalar<ntt::vector<float, 4, 4>> {
    ntt::vector<float, 4, 4> operator()(float v) const noexcept {
        ntt::vector<float, 4, 4> out;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                out(i, j) = v;
            }
        }
        return out;
    }
};

template <> struct vload_scalar<ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8> operator()(float v) const noexcept {
        ntt::vector<float, 8, 8> out;
        out(0) = _mm256_set1_ps(v);
        out(1) = _mm256_set1_ps(v);
        out(2) = _mm256_set1_ps(v);
        out(3) = _mm256_set1_ps(v);
        out(4) = _mm256_set1_ps(v);
        out(5) = _mm256_set1_ps(v);
        out(6) = _mm256_set1_ps(v);
        out(7) = _mm256_set1_ps(v);
        return out;
    }
};

template <bool AccC>
struct vmma<AccC, false, ntt::vector<float, 8, 8>, ntt::vector<float, 8, 8>,
            ntt::vector<float, 8, 8>> {
    ntt::vector<float, 8, 8>
    operator()(const ntt::vector<float, 8, 8> &lhs,
               const ntt::vector<float, 8, 8> &rhs,
               const ntt::vector<float, 8, 8> &v3) const noexcept {
        ntt::vector<float, 8, 8> output;
        for (size_t k = 0; k < 8; k++) {
            for (size_t m = 0; m < 8; m++) {
                output(m) = (k != 0 || AccC)
                                ? ntt::mul_add(lhs(m, k), rhs(k),
                                               k == 0 ? v3(m) : output(m))
                                : ntt::mul(lhs(m, k), rhs(k));
            }
        }
        return output;
    }
};
} // namespace nncase::ntt::vector_ops
