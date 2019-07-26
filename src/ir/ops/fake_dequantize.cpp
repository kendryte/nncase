#include <ir/ops/fake_dequantize.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

fake_dequantize::fake_dequantize(shape_t input_shape)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32, input_shape);
}
