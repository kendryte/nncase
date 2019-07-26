#include <ir/op_utils.h>
#include <ir/ops/reshape.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

reshape::reshape(datatype_t type, shape_t input_shape, shape_t new_shape)
    : new_shape_(normalize_reshape(input_shape, new_shape))
{
    add_input("input", type, input_shape);
    add_output("output", type, new_shape_);
}
