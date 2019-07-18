#pragma once
#include <array>
#include <optional>
#include <stdint.h>

namespace nncase
{
typedef enum _datatype
{
    dt_float32,
    dt_uint8
} datatype_t;

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
};

typedef enum _reduce_op
{
    reduce_mean,
    reduce_min,
    reduce_max
} reduce_op_t;

typedef enum _binary_op
{
    binary_add,
    binary_sub,
    binary_mul,
    binary_div
} binary_op_t;

typedef struct _quant_param
{
    int32_t zero_point;
    float scale;
} quant_param_t;

inline bool operator==(const quant_param_t &lhs, const quant_param_t &rhs) noexcept
{
    return lhs.zero_point == rhs.zero_point && lhs.scale == rhs.scale;
}

struct fixed_mul
{
    float mul;
    int8_t shift;
};

typedef enum _memory_type
{
    mem_const,
    mem_main
} memory_type_t;

using runtime_shape_t = std::array<int, 4>;
using runtime_paddings_t = std::array<padding, 4>;

struct scalar
{
    datatype_t type;
    std::array<uint8_t, 4> storage;

    scalar() = default;

    template <class T>
    scalar(T &&value) { as<T>() = value; }

    template <class T>
    T &as() noexcept { return *reinterpret_cast<T *>(storage.data()); }

    template <class T>
    const T &as() const noexcept { return *reinterpret_cast<const T *>(storage.data()); }
};

struct memory_range
{
    memory_type_t memory_type;
    datatype_t datatype;
    uint32_t start;
    uint32_t size;
};
}
