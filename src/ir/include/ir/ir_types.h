#pragma once
#include <datatypes.h>
#include <xtensor/xshape.hpp>

namespace nncase
{
namespace ir
{
    using shape_t = xt::dynamic_shape<std::size_t>;
    using axis_t = xt::dynamic_shape<int32_t>;

    inline std::string to_string(const shape_t &shape)
    {
        std::string str;
        for (size_t i = 0; i < shape.size(); i++)
        {
            str.append(std::to_string(shape[i]));
            if (i != shape.size() - 1)
                str.append("x");
        }

        return str;
    }
}
}
