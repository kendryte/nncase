#include <ir/ops/reduce.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

reduce::reduce(reduce_op_t reduce_op, shape_t input_shape, axis_t axis, float init_value, bool keep_dims)
    : reduce_op_(reduce_op), keep_dims_(keep_dims), axis_(normalize_reduce_axis(axis)), init_value_(init_value)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, get_reduced_shape(input_shape, axis_, keep_dims_));
}
