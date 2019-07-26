#include <ir/ops/reduce_window2d.h>
#include <ir/op_utils.h>

using namespace nncase;
using namespace nncase::ir;

reduce_window2d::reduce_window2d(reduce_op_t reduce_op, shape_t input_shape, float init_value, int32_t filter_h, int32_t filter_w, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation)
    : reduce_op_(reduce_op), init_value_(init_value), filter_h_(filter_h), filter_w_(filter_w), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), dilation_h_(dilation_h), dilation_w_(dilation_w), fused_activation_(fused_activation)
{
    add_input("input", dt_float32, input_shape);
    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            input_shape[1],
            get_windowed_output_size((int32_t)input_shape[2] + padding_h_.sum(), filter_h_, stride_h_, dilation_h_, false),
            get_windowed_output_size((int32_t)input_shape[3] + padding_w_.sum(), filter_w_, stride_w_, dilation_w_, false) });
}
