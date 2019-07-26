#include <ir/ops/binary.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

binary::binary(binary_op_t binary_op, shape_t input_a_shape, shape_t input_b_shape, value_range<float> fused_activation)
    : binary_op_(binary_op), fused_activation_(fused_activation)
{
    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_output("output", dt_float32, get_binary_output_shape(input_a_shape, input_b_shape));
}
