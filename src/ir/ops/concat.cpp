#include <ir/ops/concat.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

concat::concat(datatype_t type, xtl::span<shape_t> input_shapes, int32_t axis)
    : axis_(axis)
{
    if (input_shapes.empty())
        throw std::invalid_argument("there must be at least one input");

    for (size_t i = 0; i < input_shapes.size(); i++)
    {
        add_input("input_" + std::to_string(i), type, input_shapes[i]);
        concat_dims_.emplace_back(input_shapes[i][axis]);
    }

    add_output("output", type, get_concated_shape(input_shapes, axis));
}
