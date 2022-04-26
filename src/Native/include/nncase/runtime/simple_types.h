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
#include "../compiler_defs.h"
#include "../object.h"
#include "bfloat16.h"
#include "half.h"
#include "small_vector.hpp"
#include <array>

namespace nncase {
typedef enum : uint8_t {
#define DEFINE_TYPECODE(id, name, value) dt_##id = value,
#include "typecodes.def"
#undef DEFINE_TYPECODE
} typecode_t;

struct scalar {
    typecode_t type;
    std::aligned_storage_t<8> storage;

    scalar() = default;

    scalar(int8_t value) noexcept {
        type = dt_int8;
        as<int8_t>() = value;
    }

    scalar(int16_t value) noexcept {
        type = dt_int16;
        as<int16_t>() = value;
    }

    scalar(int32_t value) noexcept {
        type = dt_int32;
        as<int32_t>() = value;
    }

    scalar(uint8_t value) noexcept {
        type = dt_uint8;
        as<uint8_t>() = value;
    }

    scalar(uint16_t value) noexcept {
        type = dt_uint16;
        as<uint16_t>() = value;
    }

    scalar(uint32_t value) noexcept {
        type = dt_uint32;
        as<uint32_t>() = value;
    }

    scalar(bfloat16 value) noexcept {
        type = dt_bfloat16;
        as<bfloat16>() = value;
    }

    scalar(half value) noexcept {
        type = dt_float16;
        as<half>() = value;
    }

    scalar(float value) noexcept {
        type = dt_float32;
        as<float>() = value;
    }

    template <class T> T &as() noexcept {
        return *reinterpret_cast<T *>(&storage);
    }

    template <class T> const T &as() const noexcept {
        return *reinterpret_cast<const T *>(&storage);
    }
};

inline constexpr size_t typecode_bytes(typecode_t typecode) {
    switch (typecode) {
    case dt_uint8:
    case dt_int8:
        return 1;
    case dt_uint16:
    case dt_int16:
    case dt_float16:
    case dt_bfloat16:
        return 2;
    case dt_uint32:
    case dt_int32:
    case dt_float32:
        return 4;
    case dt_uint64:
    case dt_int64:
    case dt_float64:
        return 8;
    case dt_pointer:
        return sizeof(intptr_t);
    default:
        return -1;
    }
}

using uuid_t = std::array<uint8_t, 16>;
using dims_t = itlib::small_vector<size_t, 4>;
using strides_t = itlib::small_vector<size_t, 4>;

typedef enum _reduce_op {
    reduce_mean,
    reduce_min,
    reduce_max,
    reduce_sum
} reduce_op_t;

typedef enum _binary_op {
    binary_add,
    binary_sub,
    binary_mul,
    binary_div,
    binary_mod,
    binary_min,
    binary_max,
    binary_pow,
    binary_bitwise_and,
    binary_bitwise_or,
    binary_bitwise_xor,
    binary_logical_and,
    binary_logical_or,
    binary_logical_xor
} binary_op_t;

typedef enum _unary_op {
    unary_abs,
    unary_ceil,
    unary_cos,
    unary_exp,
    unary_floor,
    unary_log,
    unary_neg,
    unary_round,
    unary_rsqrt,
    unary_sin,
    unary_sqrt,
    unary_square,
    unary_tanh,
    unary_bitwise_not,
    unary_logical_not
} unary_op_t;

typedef enum _image_resize_mode {
    image_resize_bilinear,
    image_resize_nearest_neighbor
} image_resize_mode_t;

typedef enum _pad_mode {
    pad_constant,
    pad_reflect,
    pad_symmetric,
    pad_edge
} pad_mode_t;
} // namespace nncase
