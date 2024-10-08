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

namespace nncase::ntt {
namespace ukernels {
enum class mamtul_pack_kind {
    unknown,
    no_pack,
    pack_m,
    pack_k,
    pack_n,
    pack_mn,
    pack_mk,
    pack_kn,
    pack_mkn,
};
} // namespace ukernels

template <ukernels::mamtul_pack_kind PackKind, bool AccC, class TLhsElem,
          class TRhsElem, class TOutElem>
void u_mul_add(const TLhsElem &lhs, const TRhsElem &rhs, TOutElem &output) {
    // 1. 0D-packing
    if constexpr (PackKind == ukernels::mamtul_pack_kind::no_pack) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2. 1D-packing
    // 2.1. pack M
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_m) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2.2. pack K
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_k) {
        auto value = ntt::inner_product(lhs, rhs);
        output = AccC ? output + value : value;
    }
    // 2.3. pack N
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_n) {
        output = AccC ? ntt::mul_add(lhs, rhs, output) : ntt::mul(lhs, rhs);
    }
    // 2.4. pack M & N
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_mn ||
                       PackKind == ukernels::mamtul_pack_kind::pack_kn) {
        auto value = ntt::outer_product(lhs, rhs);
        output = AccC ? output + value : value;
    }
    // 3.1. pack MK & K
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_mk) {
        for (size_t m = 0; m < lhs.shape()[0]; m++) {
            auto value = ntt::inner_product(lhs(m), rhs);
            output(m) = AccC ? output(m) + value : value;
        }
    }
    // 3.2. pack MK & KN
    else if constexpr (PackKind == ukernels::mamtul_pack_kind::pack_mkn) {
        output = ntt::mma<AccC>(lhs, rhs, output);
    } else {
        static_assert(sizeof(TLhsElem) == 0, "Unsupported packing.");
    }
}
} // namespace nncase::ntt
