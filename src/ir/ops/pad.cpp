#include <ir/op_utils.h>
#include <ir/ops/pad.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

pad::pad(datatype_t type, shape_t input_shape, xt::svector<padding> paddings, scalar pad_value)
    : paddings_(std::move(paddings)), pad_value_(std::move(pad_value))
{
    add_input("input", type, input_shape);
    add_output("output", type, get_padded_shape(input_shape, paddings_));
}
