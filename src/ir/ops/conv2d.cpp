#include <ir/ops/conv2d.h>
#include <ir/op_utils.h>

using namespace nncase;
using namespace nncase::ir;

conv2d::conv2d(shape_t input_shape, xt::xtensor<float, 4> weights, xt::xtensor<float, 1> bias, int32_t groups, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation)
    : weights_(std::move(weights)), bias_(std::move(bias)), groups_(groups), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), dilation_h_(dilation_h), dilation_w_(dilation_w), fused_activation_(fused_activation)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            (size_t)output_channels(),
            get_windowed_output_size((int32_t)input_shape[2] + padding_h_.sum(), filter_h(), stride_h_, dilation_h_, false),
            get_windowed_output_size((int32_t)input_shape[3] + padding_w_.sum(), filter_w(), stride_w_, dilation_w_, false) });
}
