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
    class conv2d_transpose : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_conv2d_transpose);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        int32_t filter_h() const noexcept { return (int32_t)weights_.shape()[2]; }
        int32_t filter_w() const noexcept { return (int32_t)weights_.shape()[3]; }
        int32_t input_channels() const noexcept { return (int32_t)weights_.shape()[1] * groups(); }
        int32_t output_channels() const noexcept { return (int32_t)weights_.shape()[0]; }
        int32_t groups() const noexcept { return groups_; }
        padding padding_h() const noexcept { return padding_h_; }
        padding padding_w() const noexcept { return padding_w_; }
        int32_t stride_h() const noexcept { return stride_h_; }
        int32_t stride_w() const noexcept { return stride_w_; }
        int32_t dilation_h() const noexcept { return dilation_h_; }
        int32_t dilation_w() const noexcept { return dilation_w_; }
        const xt::xtensor<float, 4> &weights() const noexcept { return weights_; }
        const xt::xtensor<float, 1> &bias() const noexcept { return bias_; }
        value_range<float> fused_activation() const noexcept { return fused_activation_; }

        conv2d_transpose(shape_t input_shape, shape_t output_shape, xt::xtensor<float, 4> weights, xt::xtensor<float, 1> bias, int32_t groups, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation);

    private:
        xt::xtensor<float, 4> weights_;
        xt::xtensor<float, 1> bias_;
        int32_t groups_;
        padding padding_h_;
        padding padding_w_;
        int32_t stride_h_;
        int32_t stride_w_;
        int32_t dilation_h_;
        int32_t dilation_w_;
        value_range<float> fused_activation_;
    };

    class quantized_conv2d_transpose : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_quantized_conv2d);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        int32_t filter_h() const noexcept { return (int32_t)weights_.shape()[2]; }
        int32_t filter_w() const noexcept { return (int32_t)weights_.shape()[3]; }
        int32_t input_channels() const noexcept { return (int32_t)weights_.shape()[1] * groups(); }
        int32_t output_channels() const noexcept { return (int32_t)weights_.shape()[0]; }
        int32_t groups() const noexcept { return groups_; }
        padding padding_h() const noexcept { return padding_h_; }
        padding padding_w() const noexcept { return padding_w_; }
        int32_t stride_h() const noexcept { return stride_h_; }
        int32_t stride_w() const noexcept { return stride_w_; }
        int32_t dilation_h() const noexcept { return dilation_h_; }
        int32_t dilation_w() const noexcept { return dilation_w_; }
        int32_t input_offset() const noexcept { return input_offset_; }
        int32_t filter_offset() const noexcept { return filter_offset_; }
        int32_t output_mul() const noexcept { return output_mul_; }
        int32_t output_shift() const noexcept { return output_shift_; }
        int32_t output_offset() const noexcept { return output_offset_; }
        const xt::xtensor<uint8_t, 4> &weights() const noexcept { return weights_; }
        const xt::xtensor<int32_t, 1> &bias() const noexcept { return bias_; }

        quantized_conv2d_transpose(shape_t input_shape, xt::xtensor<uint8_t, 4> weights, xt::xtensor<int32_t, 1> bias, int32_t groups, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h,
            int32_t dilation_w, int32_t input_offset, int32_t filter_offset, int32_t output_mul, int32_t output_shift, int32_t output_offset);

    private:
        xt::xtensor<uint8_t, 4> weights_;
        xt::xtensor<int32_t, 1> bias_;
        int32_t groups_;
        padding padding_h_;
        padding padding_w_;
        int32_t stride_h_;
        int32_t stride_w_;
        int32_t dilation_h_;
        int32_t dilation_w_;
        int32_t input_offset_;
        int32_t filter_offset_;
        int32_t output_mul_;
        int32_t output_shift_;
        int32_t output_offset_;
    };
}
}
