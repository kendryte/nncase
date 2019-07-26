#include <ir/ops/dequantize.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

dequantize::dequantize(shape_t input_shape, quant_param_t quant_param)
    : quant_param_(quant_param)
{
    add_input("input", dt_uint8, input_shape);
    add_output("output", dt_float32, input_shape);
}
