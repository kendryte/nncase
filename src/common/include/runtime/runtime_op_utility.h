#pragma once
#include "../datatypes.h"

namespace nncase
{
namespace runtime
{
    inline size_t get_bytes(datatype_t type)
    {
        size_t element_size;

        switch (type)
        {
        case dt_float32:
            element_size = 4;
            break;
        case dt_uint8:
            element_size = 1;
            break;
        default:
            NNCASE_THROW(std::runtime_error, "Not supported data type");
        }

        return element_size;
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

    template <class T>
    T carry_shift(T value, uint8_t shift)
    {
        if (shift > 0)
        {
            value >>= shift - 1;
            if (value & 0x1)
            {
                if (value < 0)
                    value = (value >> 1) - 1;
                else
                    value = (value >> 1) + 1;
            }
            else
            {
                value >>= 1;
            }
        }

        return value;
    }

    inline int32_t mul_and_carry_shift(int32_t value, int32_t mul, uint8_t shift)
    {
        return (int32_t)carry_shift((int64_t)value * mul, shift);
    }

    template <class T>
    struct to_datatype
    {
    };

    template <>
    struct to_datatype<float>
    {
        static constexpr datatype_t type = dt_float32;
    };

    template <>
    struct to_datatype<uint8_t>
    {
        static constexpr datatype_t type = dt_uint8;
    };

    template <class T>
    inline constexpr datatype_t to_datatype_v = to_datatype<T>::type;
}
}
