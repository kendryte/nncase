/* Copyright 2019 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
