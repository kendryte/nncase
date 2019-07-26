#include <ir/ops/matmul.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

matmul::matmul(shape_t input_a_shape, shape_t input_b_shape, xt::xtensor<float, 1> bias, value_range<float> fused_activation)
    : bias_(std::move(bias)), fused_activation_(fused_activation)
{
    if (input_a_shape.size() != 2 || input_b_shape.size() != 2)
        throw std::invalid_argument("inputs must be 2 rank");
    if (input_a_shape[1] != input_b_shape[0])
        throw std::invalid_argument("input a's cols must be equal to input b's rows");

    add_input("input_a", dt_float32, input_a_shape);
    add_input("input_b", dt_float32, input_b_shape);
    add_output("output", dt_float32, shape_t { input_a_shape[0], input_b_shape[1] });
}
