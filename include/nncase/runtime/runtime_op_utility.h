/* Copyright 2020 Canaan Inc.
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
#include "datatypes.h"
#include "result.h"
#include <xtensor/xstrides.hpp>

BEGIN_NS_NNCASE_RUNTIME

inline constexpr size_t get_bytes(datatype_t type)
{
    switch (type)
    {
#define DEFINE_DATATYPE(id, t, name, value) \
    case (dt_##id):                         \
        return sizeof(t);
#include <nncase/runtime/datatypes.def>
#undef DEFINE_DATATYPE
    default:
        return -1;
    }
}

inline size_t get_bytes(datatype_t type, const runtime_shape_t &shape)
{
    return xt::compute_size(shape) * get_bytes(type);
}

inline size_t compute_size(const runtime_shape_t &shape, const runtime_shape_t &strides)
{
    size_t data_size = 0;
    for (size_t i = 0; i < shape.size(); i++)
        data_size += (shape[i] - 1) * strides[i];
    data_size += strides.back();
    return data_size;
}

inline size_t get_bytes(datatype_t type, const runtime_shape_t &shape, const runtime_shape_t &strides)
{
    return compute_size(shape, strides) * get_bytes(type);
}

inline runtime_shape_t get_default_strides(const runtime_shape_t &shape)
{
    runtime_shape_t strides(shape.size());
    xt::compute_strides(shape, xt::layout_type::row_major, strides);
    return strides;
}

template <class TShape>
TShape convert_shape_type(const TShape &shape, datatype_t src, datatype_t dest)
{
    const auto src_size = get_bytes(src);
    const auto dest_size = get_bytes(dest);

    TShape new_shape = shape;
    if (!new_shape.empty())
    {
        auto &v = new_shape.back();
        v = new_shape.back() * src_size / dest_size;
    }

    return new_shape;
}

template <class TShape>
result<TShape> convert_strides_type(const TShape &strides, datatype_t src, datatype_t dest)
{
    const auto src_size = get_bytes(src);
    const auto dest_size = get_bytes(dest);

    if (src_size == dest_size)
        return ok(strides);

    TShape new_strides = strides;
    // 1. Except last dim
    for (size_t i = 0; i < new_strides.size() - 1; i++)
    {
        auto &v = new_strides[i];
        if (v == 0)
            v = 1;
        v = v * src_size / dest_size;
    }

    // 2. Last dim
    if (!new_strides.empty())
    {
        // 2.1. If last dim is not 0 or 1, unsupported
        auto last_dim = new_strides.back();
        if (last_dim != 0 || last_dim != 1)
            return err(std::errc::not_supported);
    }

    return ok(new_strides);
}

template <int32_t Bits, class T>
uint8_t count_leading_zeros(T value)
{
    uint8_t num_zeroes = 0;
    for (int32_t i = Bits - 1; i >= 0; i--)
    {
        if ((value & (1ULL << i)) == 0)
            ++num_zeroes;
        else
            break;
    }

    return num_zeroes;
}

template <class T = uint64_t>
inline T bit_mask(uint8_t shift)
{
    return (T(1) << shift) - 1;
}

template <class T, bool Banker = false>
T carry_shift(T value, int32_t shift)
{
    if (shift > 0)
    {
        if constexpr (Banker)
        {
            T result;
            // Sign |  Int (T - shift - 1 bits) | Frac (shift bits)
            //  S      IIII                       FFF
            auto integral = value >> shift;
            auto fractional = value & bit_mask(shift);
            auto sign = value < 0 ? -1 : 1;
            auto half = size_t(1) << (shift - 1);

            // frac < 0.5
            if (fractional < half)
            {
                return integral;
            }
            // frac > 0.5
            else if (fractional > half)
            {
                return integral + sign;
            }
            // frac == 0.5
            else
            {
                // odd
                if (integral & 1)
                    return integral + sign;
                // even
                else
                    return integral;
            }

            return result;
        }
        else
        {
            value += T(1) << (shift - 1);
            value >>= shift;
        }
    }
    else if (shift < 0)
    {
        value = value << (-shift);
    }

    return value;
}

template <bool Banker = false>
inline int32_t mul_and_carry_shift(int32_t value, int32_t mul, int32_t shift)
{
    return (int32_t)carry_shift<int64_t, Banker>((int64_t)value * mul, shift);
}

template <uint8_t Bits>
inline int32_t clamp(int32_t value)
{
    auto min = std::numeric_limits<int32_t>::lowest() >> (32 - Bits);
    auto max = std::numeric_limits<int32_t>::max() >> (32 - Bits);
    return std::clamp(value, min, max);
}

END_NS_NNCASE_RUNTIME
