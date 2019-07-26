#include <ir/ops/quantize.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

quantize::quantize(shape_t input_shape, quant_param_t quant_param)
    : quant_param_(quant_param)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_uint8, input_shape);
}
