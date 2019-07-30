#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class reduce_window2d : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_reduce_window2d);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        reduce_op_t reduce_op() const noexcept { return reduce_op_; }
        float init_value() const noexcept { return init_value_; }
        int32_t filter_h() const noexcept { return filter_h_; }
        int32_t filter_w() const noexcept { return filter_w_; }
        padding padding_h() const noexcept { return padding_h_; }
        padding padding_w() const noexcept { return padding_w_; }
        int32_t stride_h() const noexcept { return stride_h_; }
        int32_t stride_w() const noexcept { return stride_w_; }
        int32_t dilation_h() const noexcept { return dilation_h_; }
        int32_t dilation_w() const noexcept { return dilation_w_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        reduce_window2d(reduce_op_t reduce_op, shape_t input_shape, float init_value, int32_t filter_h, int32_t filter_w, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation);

    private:
        reduce_op_t reduce_op_;
        float init_value_;
        int32_t filter_h_;
        int32_t filter_w_;
        padding padding_h_;
        padding padding_w_;
        int32_t stride_h_;
        int32_t stride_w_;
        int32_t dilation_h_;
        int32_t dilation_w_;
        value_range<float> fused_activation_;
    };
}
}
