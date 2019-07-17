#pragma once
#include <cassert>
#include <datatypes.h>

namespace nncase
{
namespace runtime
{
    inline size_t get_bytes(datatype type)
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
            assert(!"Not supported data type");
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
}
}
