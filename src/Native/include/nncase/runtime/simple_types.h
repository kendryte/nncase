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
    case dt_boolean:
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

struct padding {
    int32_t before;
    int32_t after;
    int32_t interior = 0;

    int32_t sum() const noexcept { return before + after; }

    static padding zero() noexcept { return {}; }
};

using uuid_t = std::array<uint8_t, 16>;
using dims_t = itlib::small_vector<size_t, 8>;
using axes_t = itlib::small_vector<int64_t, 8>;
using strides_t = itlib::small_vector<size_t, 8>;
using paddings_t = itlib::small_vector<padding, 4>;

template <class... Ints>
auto fixed_dims(Ints &&...values) -> std::array<size_t, sizeof...(Ints)> {
    return {(size_t)values...};
}

template <class T> struct value_range {
    T min;
    T max;

    static constexpr value_range<T> full() noexcept {
        if (std::is_floating_point<T>::value ||
            std::is_same<T, bfloat16>::value || std::is_same<T, half>::value)
            return {-std::numeric_limits<T>::infinity(),
                    std::numeric_limits<T>::infinity()};
        else
            return {std::numeric_limits<T>::lowest(),
                    std::numeric_limits<T>::max()};
    }

    static constexpr value_range<T> nonnegative() noexcept {
        return {0, std::numeric_limits<T>::max()};
    }

    constexpr T length() const noexcept { return max - min; }
};

typedef struct _quant_param {
    int32_t zero_point;
    float scale;

    template <class T> constexpr value_range<float> range() const noexcept {
        return {(std::numeric_limits<T>::lowest() - zero_point) * scale,
                (std::numeric_limits<T>::max() - zero_point) * scale};
    }
} quant_param_t;

inline bool operator==(const quant_param_t &lhs,
                       const quant_param_t &rhs) noexcept {
    return lhs.zero_point == rhs.zero_point && lhs.scale == rhs.scale;
}

inline bool almost_equal(const quant_param_t &lhs,
                         const quant_param_t &rhs) noexcept {
    return lhs.zero_point == rhs.zero_point &&
           fabs(lhs.scale - rhs.scale) <= std::numeric_limits<float>::epsilon();
}

namespace runtime {
typedef enum sync_op_ { sync_invalidate, sync_write_back } sync_op_t;

typedef enum map_access_ {
    map_none = 0,
    map_read = 1,
    map_write = 2,
    map_read_write = 3
} map_access_t;

DEFINE_ENUM_BITMASK_OPERATORS(map_access_t)

enum class host_sync_status_t { valid, need_invalidate, need_write_back };
} // namespace runtime
} // namespace nncase
