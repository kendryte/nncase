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
#include "../primitive_ops.h"
#include "../vector_ops.h"

namespace nncase::ntt {
namespace ukernels {
enum class matmul_vectorize_kind {
    unknown,
    no_vectorize,
    vectorize_m,
    vectorize_k,
    vectorize_n,
    vectorize_mn,
    vectorize_mk,
    vectorize_kn,
    vectorize_mkn,
};
} // namespace ukernels

template <ukernels::matmul_vectorize_kind VectorizeKind, bool AccC, class TLhsElem,
          class TRhsElem, class TOutElem>
void u_mul_add(const TLhsElem &lhs, const TRhsElem &rhs, TOutElem &output) {
    // 1. 0D-vectorize
    if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::no_vectorize) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2. 1D-vectorize
    // 2.1. vectorize M
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_m) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2.2. vectorize K
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_k) {
        auto value = ntt::inner_product(lhs, rhs);
        output = AccC ? output + value : value;
    }
    // 2.3. vectorize N
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_n) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2.4. vectorize M & N
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_mn ||
                       VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_kn) {
        auto value = ntt::outer_product(lhs, rhs);
        output = AccC ? output + value : value;
    }
    // 3.1. vectorize MK & K
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_mk) {
        for (size_t m = 0; m < lhs.shape()[0]; m++) {
            auto value = ntt::inner_product(lhs(m), rhs);
            output(m) = AccC ? output(m) + value : value;
        }
    }
    // 3.2. vectorize MK & KN
    else if constexpr (VectorizeKind == ukernels::matmul_vectorize_kind::vectorize_mkn) {
        output = ntt::vmma<AccC, false>(lhs, rhs, output);
    } else {
        static_assert(sizeof(TLhsElem) == 0, "Unsupported vectorize.");
    }
}
} // namespace nncase::ntt
