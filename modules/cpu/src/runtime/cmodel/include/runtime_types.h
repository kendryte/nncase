#pragma once
#include "small_vector.hpp"

using dims_t = itlib::small_vector<size_t, 8>;
using strides_t = itlib::small_vector<size_t, 8>;
using axes_t = itlib::small_vector<int64_t, 8>;

enum class image_resize_mode_t : uint8_t {
    bilinear = 0,
    nearest_neighbor = 1,
};

enum class image_resize_transformation_mode_t : int32_t {
    half_pixel = 0,
    pytorch_half_pixel = 1,
    align_corners = 2,
    asymmetric = 3,
    tfcrop_and_resize = 4,
};

enum class image_resize_nearest_mode_t : int32_t {
    round_prefer_floor = 0,
    round_prefer_ceil = 1,
    floor = 2,
    ceil = 3,
};

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
    swish = 22
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
    idenity_a = 16,
};