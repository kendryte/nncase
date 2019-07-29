#pragma once
#include <datatypes.h>
#include <xtensor/xshape.hpp>

namespace nncase
{
namespace ir
{
    using shape_t = xt::dynamic_shape<std::size_t>;
    using axis_t = xt::dynamic_shape<int32_t>;
}
}
