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
#include "vector.h"

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, ...)        \
    namespace nncase::ntt {                                                    \
    template <> struct vector_storage_traits<element_type_, __VA_ARGS__> {     \
        using buffer_type = native_type;                                       \
        using element_type = element_type_;

#define NTT_END_DEFINE_NATIVE_VECTOR()                                         \
    }                                                                          \
    ;                                                                          \
    }

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT(element_type_, native_type,     \
                                               ...)                            \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__)    \
                                                                               \
    template <Dimensions TIndex>                                               \
    static element_type_ get_element(const native_type &array,                 \
                                     const TIndex &index) noexcept {           \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        return array[index[fixed_dim_zero]];                                   \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_type &array, const TIndex &index,           \
                            element_type_ value) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        array[index[fixed_dim_zero]] = value;                                  \
    }

#define NTT_BEGIN_DEFINE_NATIVE_VECTOR_DEFAULT_CAST(                           \
    element_type_, native_type, cast_type, ...)                                \
    NTT_BEGIN_DEFINE_NATIVE_VECTOR(element_type_, native_type, __VA_ARGS__)    \
                                                                               \
    template <Dimensions TIndex>                                               \
    static element_type_ get_element(const native_type &array,                 \
                                     const TIndex &index) noexcept {           \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        return reinterpret_cast<const cast_type &>(                            \
            array)[index[fixed_dim_zero]];                                     \
    }                                                                          \
                                                                               \
    template <Dimensions TIndex>                                               \
    static void set_element(native_type &array, const TIndex &index,           \
                            element_type_ value) noexcept {                    \
        static_assert(TIndex::rank() == 1, "index must be 1D");                \
        reinterpret_cast<cast_type &>(array)[index[fixed_dim_zero]] = value;   \
    }
