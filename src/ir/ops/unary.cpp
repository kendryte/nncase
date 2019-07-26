#include <ir/ops/unary.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

unary::unary(unary_op_t unary_op, shape_t input_shape)
    : unary_op_(unary_op)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, input_shape);
}
