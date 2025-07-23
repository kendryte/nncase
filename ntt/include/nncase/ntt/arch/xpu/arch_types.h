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
#include <array>

#ifndef NTT_VLEN
#define NTT_VLEN 1024
#endif

#ifndef SYS_MODE
// Simulation mode

// mask vectors
#define NTT_DEFINE_NATIVE_MASK_VECTOR(element_type, bits)                      \
    using native_mask_b##bits##_t = std::array<element_type, NTT_VLEN / bits>; \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(bool, native_mask_b##bits##_t,              \
                                   NTT_VLEN / bits)                            \
                                                                               \
    template <Dimensions TIndex>                                               \
    static bool get_element(const native_mask_b##bits##_t &array,              \
                            const TIndex &index) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const auto offset = (size_t)index[dim_zero];                           \
        return array[offset] != 0;                                             \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_mask_b##bits##_t &array,                    \
                            const TIndex &index, bool value) noexcept {        \
        constexpr auto true_value = element_type(-1);                          \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        const auto offset = (size_t)index[dim_zero];                           \
        array[offset] = value ? true_value : 0;                                \
    }                                                                          \
    NTT_END_DEFINE_NATIVE_VECTOR()

NTT_DEFINE_NATIVE_MASK_VECTOR(uint8_t, 8)
NTT_DEFINE_NATIVE_MASK_VECTOR(uint16_t, 16)
NTT_DEFINE_NATIVE_MASK_VECTOR(uint32_t, 32)
NTT_DEFINE_NATIVE_MASK_VECTOR(uint64_t, 64)

#undef NTT_DEFINE_NATIVE_MASK_VECTOR
#endif
