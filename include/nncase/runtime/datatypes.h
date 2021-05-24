/* Copyright 2019-2020 Canaan Inc.
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
#include "bfloat16.h"
#include "compiler_defs.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <xtensor/xshape.hpp>

namespace nncase
{
typedef enum _datatype : uint8_t
{
#define DEFINE_DATATYPE(id, t, name, value) dt_##id = value,
#include "datatypes.def"
#undef DEFINE_DATATYPE
} datatype_t;

namespace detail
{
    template <datatype_t Type>
    struct datatype_to_cpp_type
    {
    };

    template <class T>
    struct cpp_type_to_datatype
    {
    };

#if NNCASE_HAVE_STD_BYTE
    template <>
    struct cpp_type_to_datatype<std::byte>
    {
        static constexpr datatype_t type = dt_uint8;
    };
#endif

#define DEFINE_DATATYPE(id, t, name, value)         \
    template <>                                     \
    struct datatype_to_cpp_type<dt_##id>            \
    {                                               \
        using type = t;                             \
    };                                              \
    template <>                                     \
    struct cpp_type_to_datatype<t>                  \
    {                                               \
        static constexpr datatype_t type = dt_##id; \
    };
#include "datatypes.def"
#undef DEFINE_DATATYPE
}

template <class T>
constexpr datatype_t to_datatype() noexcept
{
    return detail::cpp_type_to_datatype<T>::type;
}

template <datatype_t Type>
using to_cpp_type_t = typename detail::datatype_to_cpp_type<Type>::type;

struct padding
{
    int32_t before;
    int32_t after;

    int32_t sum() const noexcept { return before + after; }

    static padding zero() noexcept { return {}; }
};

template <class T>
struct value_range
{
    T min;
    T max;

    static constexpr value_range<T> full() noexcept
    {
        if (std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value)
            return { -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity() };
        else
            return { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max() };
    }

    static constexpr value_range<T> nonnegative() noexcept
    {
        return { 0, std::numeric_limits<T>::max() };
    }
};

typedef enum _reduce_op
{
    reduce_mean,
    reduce_min,
    reduce_max,
    reduce_sum
} reduce_op_t;

typedef enum _binary_op
{
    binary_add,
    binary_sub,
    binary_mul,
    binary_div,
    binary_min,
    binary_max,
    binary_pow,
    binary_floor_div,
    binary_floor_mod,
    binary_bitwise_and,
    binary_bitwise_or,
    binary_bitwise_xor,
    binary_logical_and,
    binary_logical_or,
    binary_logical_xor
} binary_op_t;

inline std::string binary_op_to_string(binary_op_t op)
{
    switch (op)
    {
    case binary_add:
        return "binary_add";
    case binary_sub:
        return "binary_sub";
    case binary_mul:
        return "binary_mul";
    case binary_div:
        return "binary_div";
    case binary_min:
        return "binary_min";
    case binary_max:
        return "binary_max";
    case binary_pow:
        return "binary_pow";
    case binary_floor_div:
        return "binary_floor_div";
    case binary_floor_mod:
        return "binary_floor_mod";
    case binary_bitwise_and:
        return "binary_bitwise_and";
    case binary_bitwise_or:
        return "binary_bitwise_or";
    case binary_bitwise_xor:
        return "binary_bitwise_xor";
    case binary_logical_and:
        return "binary_logical_and";
    case binary_logical_or:
        return "binary_logical_or";
    case binary_logical_xor:
        return "binary_logical_xor";
    }
    return "unknown";
}

typedef enum _unary_op
{
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

inline std::string unary_op_to_string(unary_op_t op)
{
    switch (op)
    {
    case unary_abs:
        return "unary_abs";
    case unary_ceil:
        return "unary_ceil";
    case unary_cos:
        return "unary_cos";
    case unary_exp:
        return "unary_exp";
    case unary_floor:
        return "unary_floor";
    case unary_log:
        return "unary_log";
    case unary_neg:
        return "unary_neg";
    case unary_round:
        return "unary_round";
    case unary_rsqrt:
        return "unary_rsqrt";
    case unary_sin:
        return "unary_sin";
    case unary_sqrt:
        return "unary_sqrt";
    case unary_square:
        return "unary_square";
    case unary_tanh:
        return "unary_tanh";
    case unary_bitwise_not:
        return "unary_bitwise_not";
    case unary_logical_not:
        return "unary_logical_not";
    }
    return "unknown";
}

typedef enum _image_resize_mode
{
    image_resize_bilinear,
    image_resize_nearest_neighbor
} image_resize_mode_t;

typedef enum _pad_mode
{
    pad_constant,
    pad_reflect,
    pad_symmetric,
    pad_edge
} pad_mode_t;

typedef struct _quant_param
{
    int32_t zero_point;
    float scale;
} quant_param_t;

inline bool operator==(const quant_param_t &lhs, const quant_param_t &rhs) noexcept
{
    return lhs.zero_point == rhs.zero_point && lhs.scale == rhs.scale;
}

inline bool almost_equal(const quant_param_t &lhs, const quant_param_t &rhs) noexcept
{
    return lhs.zero_point == rhs.zero_point
        && fabs(lhs.scale - rhs.scale) <= std::numeric_limits<float>::epsilon();
}

struct fixed_mul
{
    float mul;
    int8_t shift;

    int32_t rounded_mul() const noexcept { return (int32_t)lrintf(mul); }
};

using memory_location_t = uint8_t;
NNCASE_INLINE_VAR constexpr memory_location_t mem_input = 0;
NNCASE_INLINE_VAR constexpr memory_location_t mem_output = 1;
NNCASE_INLINE_VAR constexpr memory_location_t mem_rdata = 2;
NNCASE_INLINE_VAR constexpr memory_location_t mem_data = 3;

using runtime_shape_t = xt::svector<size_t, 4>;
using runtime_axis_t = xt::svector<int32_t, 4>;
using runtime_paddings_t = xt::svector<padding, 4>;

struct alignas(8) scalar
{
    datatype_t type;
    std::array<uint8_t, 8> storage;

    scalar() = default;

    scalar(int8_t value) noexcept
    {
        type = dt_int8;
        as<int8_t>() = value;
    }

    scalar(int16_t value) noexcept
    {
        type = dt_int16;
        as<int16_t>() = value;
    }

    scalar(int32_t value) noexcept
    {
        type = dt_int32;
        as<int32_t>() = value;
    }

    scalar(uint8_t value) noexcept
    {
        type = dt_uint8;
        as<uint8_t>() = value;
    }

    scalar(uint16_t value) noexcept
    {
        type = dt_uint16;
        as<uint16_t>() = value;
    }

    scalar(uint32_t value) noexcept
    {
        type = dt_uint32;
        as<uint32_t>() = value;
    }

    scalar(bfloat16 value) noexcept
    {
        type = dt_bfloat16;
        as<bfloat16>() = value;
    }

    scalar(float value) noexcept
    {
        type = dt_float32;
        as<float>() = value;
    }

    template <class T>
    T &as() noexcept { return *reinterpret_cast<T *>(storage.data()); }

    template <class T>
    const T &as() const noexcept { return *reinterpret_cast<const T *>(storage.data()); }
};

struct memory_range
{
    memory_location_t memory_location;
    datatype_t datatype;
    uint16_t reserved0;
    uint32_t start;
    uint32_t size;
};

NNCASE_INLINE_VAR constexpr size_t MAX_MODULE_TYPE_LENGTH = 16;

typedef std::array<char, MAX_MODULE_TYPE_LENGTH> module_type_t;

template <std::size_t N, std::size_t... Is>
constexpr module_type_t
to_module_type(const char (&a)[N], std::index_sequence<Is...>)
{
    return { { a[Is]... } };
}

template <std::size_t N>
constexpr module_type_t to_module_type(const char (&a)[N])
{
    return to_module_type(a, std::make_index_sequence<N>());
}

inline padding operator+(const padding &lhs, const padding &rhs) noexcept
{
    return { lhs.before + rhs.before, lhs.after + rhs.after };
}

inline bool operator==(const padding &lhs, const padding &rhs) noexcept
{
    return lhs.before == rhs.before && lhs.after == rhs.after;
}

inline bool operator!=(const padding &lhs, const padding &rhs) noexcept
{
    return lhs.before != rhs.before || lhs.after != rhs.after;
}

template <class T>
bool operator==(const value_range<T> &lhs, const value_range<T> &rhs) noexcept
{
    return lhs.min == rhs.min && lhs.max == rhs.max;
}

template <class T>
bool operator!=(const value_range<T> &lhs, const value_range<T> &rhs) noexcept
{
    return lhs.min != rhs.min || lhs.max != rhs.max;
}

inline bool operator==(const scalar &lhs, const scalar &rhs) noexcept
{
    return lhs.type == rhs.type && lhs.storage == rhs.storage;
}

inline bool operator!=(const scalar &lhs, const scalar &rhs) noexcept
{
    return lhs.type != rhs.type || lhs.storage != rhs.storage;
}
}
