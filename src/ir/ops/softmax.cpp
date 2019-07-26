#include <ir/ops/softmax.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

softmax::softmax(shape_t input_shape, float beta)
    : beta_(beta)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, input_shape);
}
