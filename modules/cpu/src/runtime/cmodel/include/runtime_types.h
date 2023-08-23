#pragma once
#include <nncase/runtime/small_vector.hpp>

using dims_t = itlib::small_vector<size_t, 8>;
using strides_t = itlib::small_vector<size_t, 8>;
using axes_t = itlib::small_vector<int64_t, 8>;

enum class unary_op_t : uint8_t {
    abs = 0,
    acos = 1,
    acosh = 2,
    asin = 3,
    asinh = 4,
    ceil = 5,
    cos = 6,
    cosh = 7,
    exp = 8,
    floor = 9,
    log = 10,
    neg = 11,
    round = 12,
    rsqrt = 13,
    sin = 14,
    sinh = 15,
    sign = 16,
    sqrt = 17,
    square = 18,
    tanh = 19,
    bitwise_not = 20,
    logical_not = 21,
};

enum class binary_op_t : uint8_t {
    add = 0,
    sub = 1,
    mul = 2,
    div = 3,
    mod = 4,
    min = 5,
    max = 6,
    pow = 7,
    bitwise_and = 8,
    bitwise_or = 9,
    bitwise_xor = 10,
    logical_and = 11,
    logical_or = 12,
    logical_xor = 13,
    left_shift = 14,
    right_shift = 15,
};