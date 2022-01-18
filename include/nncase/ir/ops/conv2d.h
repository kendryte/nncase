/* Copyright 2019-2021 Canaan Inc.
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
#include <xtensor/xarray.hpp>

namespace nncase::ir
{
class NNCASE_API conv2d : public node
{
public:
    DEFINE_NODE_OPCODE(op_conv2d);

    const input_connector &weights() const { return input_at(1); }

    input_connector &input() { return input_at(0); }
    input_connector &weights() { return input_at(1); }
    input_connector &bias() { return input_at(2); }
    output_connector &output() { return output_at(0); }

    int32_t filter_h() const noexcept { return (int32_t)weights().shape()[2]; }
    int32_t filter_w() const noexcept { return (int32_t)weights().shape()[3]; }
    int32_t input_channels() const noexcept { return (int32_t)weights().shape()[1] * groups(); }
    int32_t output_channels() const noexcept { return (int32_t)weights().shape()[0]; }
    int32_t groups() const noexcept { return groups_; }
    bool is_depthwise() const noexcept { return input_channels() == output_channels() && output_channels() == groups() && groups() != 1; }
    padding padding_h() const noexcept { return padding_h_; }
    padding padding_w() const noexcept { return padding_w_; }
    int32_t stride_h() const noexcept { return stride_h_; }
    int32_t stride_w() const noexcept { return stride_w_; }
    int32_t dilation_h() const noexcept { return dilation_h_; }
    int32_t dilation_w() const noexcept { return dilation_w_; }
    value_range<float> fused_activation() const noexcept { return fused_activation_; }

    conv2d(shape_t input_shape, shape_t weights_shape, int32_t groups, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation);

protected:
    bool properties_equal(node &other) const override;

private:
    int32_t groups_;
    padding padding_h_;
    padding padding_w_;
    int32_t stride_h_;
    int32_t stride_w_;
    int32_t dilation_h_;
    int32_t dilation_w_;
    value_range<float> fused_activation_;
};
}
