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
#include "../tflite_importer.h"
#include <ir/ops/conv2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &options = *op.builtin_options_as_Conv2DOptions();

    auto weights_tensor = xt::transpose(dequantize_tensor<4>(weights), { 0, 3, 1, 2 });
    auto bias_tensor = load_tensor<float, 1>(bias);

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));

    auto in_h = pre_trans->output().shape()[2];
    auto in_w = pre_trans->output().shape()[3];
    auto f_h = weights_tensor.shape()[2];
    auto f_w = weights_tensor.shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto conv = graph_.emplace<conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), 1,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
    conv->input().connect(pre_trans->output());

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->input().connect(conv->output());

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}

DEFINE_TFLITE_LOWER(DEPTHWISE_CONV_2D)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &weights = get_tensor(op.inputs(), 1);
    auto &bias = get_tensor(op.inputs(), 2);
    auto &options = *op.builtin_options_as_DepthwiseConv2DOptions();
    auto opname = weights.name()->string_view().substr(0, weights.name()->string_view().find_first_of('/'));

    auto weights_tensor = xt::transpose(dequantize_tensor<4>(weights), { 3, 0, 1, 2 });
    auto bias_tensor = load_tensor<float, 1>(bias);

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));

    auto in_h = pre_trans->output().shape()[2];
    auto in_w = pre_trans->output().shape()[3];
    auto groups = weights_tensor.shape()[0];
    auto f_h = weights_tensor.shape()[2];
    auto f_w = weights_tensor.shape()[3];
    auto stride_h = options.stride_h();
    auto stride_w = options.stride_w();
    auto dilation_h = options.dilation_h_factor();
    auto dilation_w = options.dilation_w_factor();
    auto pad_h = get_windowed_padding(in_h, f_h, stride_h, dilation_h, options.padding() == tflite::Padding_SAME);
    auto pad_w = get_windowed_padding(in_w, f_w, stride_w, dilation_w, options.padding() == tflite::Padding_SAME);
    auto depth_mul = options.depth_multiplier();
    conv2d *conv;
    if (pre_trans->output().shape()[1] == 1 && depth_mul == groups)
    {
        conv = graph_.emplace<conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), 1,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
    }
    else if (depth_mul != 1)
    {
        throw std::runtime_error("DepthwiseConv2d " + std::string(opname) + " with depth_multiplier " + std::to_string(depth_mul) + " is not supported");
    }
    else
    {
        conv = graph_.emplace<conv2d>(pre_trans->output().shape(), std::move(weights_tensor), std::move(bias_tensor), groups,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, to_float_clamp_range(options.fused_activation_function()));
    }

    conv->input().connect(pre_trans->output());

    auto sur_trans = nchw_to_nhwc(dt_float32, conv->output().shape());
    sur_trans->input().connect(conv->output());

    input_tensors_.emplace(&pre_trans->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &sur_trans->output());
}