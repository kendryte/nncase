#pragma once
#include "../datatypes.h"

namespace nncase
{
namespace runtime
{
    constexpr std::size_t get_bytes(datatype type)
    {
        switch (type)
        {
        case dt_float32:
            return 4;
        case dt_uint8:
            return 1;
        default:
            return 0;
        }
    }
}
}
