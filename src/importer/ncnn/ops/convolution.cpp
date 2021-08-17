// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "../ncnn_importer.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

ir::node* nncase::importer::ncnn_importer::convert_op_Convolution(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin& mb)
{
    return convert_op_ConvolutionDepthWise(layer, pd, mb);
}

ir::node* nncase::importer::ncnn_importer::convert_op_ConvolutionDepthWise(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin& mb)
{
    const int num_output = pd.get(0, 0);
    const int kernel_w = pd.get(1, 0);
    const int kernel_h = pd.get(11, kernel_w);
    const int dilation_w = pd.get(2, 1);
    const int dilation_h = pd.get(12, dilation_w);
    const int stride_w = pd.get(3, 1);
    const int stride_h = pd.get(13, stride_w);
    const int pad_left = pd.get(4, 0);
    const int pad_right = pd.get(15, pad_left);
    const int pad_top = pd.get(14, pad_left);
    const int pad_bottom = pd.get(16, pad_top);
    const int pad_value = pd.get(18, 0.f);
    const int bias_term = pd.get(5, 0);
    const int weight_data_size = pd.get(6, 0);
    const int group = pd.get(7, 1);
    const int activation_type = pd.get(9, 0);
    const ncnn::Mat activation_params = pd.get(10, ncnn::Mat());

    const int num_input = weight_data_size / num_output / kernel_w / kernel_h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const auto &op_name = layer.name;

    auto in_shape = layer.bottom_shapes[0];

    const shape_t weight_shape = { (size_t)num_output, (size_t)num_input, (size_t)kernel_h, (size_t)kernel_w };
    const shape_t bias_shape = { (size_t)num_output };

    padding padding_h;
    padding padding_w;
    {
        if (pad_left == -233) // SAME_UPPER
        {
            const int w = in_shape[2];
            const int h = in_shape[1];
            const int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;

            padding_h = {hpad / 2, hpad - hpad / 2};
            padding_w = {wpad / 2, wpad - wpad / 2};
        }
        else if (pad_left == -234) // SAME_LOWER
        {
            const int w = in_shape[2];
            const int h = in_shape[1];
            const int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;

            padding_h = {hpad - hpad / 2, hpad / 2};
            padding_w = {wpad - wpad / 2, wpad / 2};
        }
        else
        {
            padding_h = {pad_top, pad_bottom};
            padding_w = {pad_left, pad_right};
        }
    }

    value_range<float> fused_activation = value_range<float>::full();
    {
        if (activation_type == 1)
        {
            fused_activation = {0.f, INFINITY};
        }
        if (activation_type == 3)
        {
            fused_activation = {activation_params[0], activation_params[1]};
        }
    }

    ir::conv2d* conv_op = 0;
    {
        if (pad_value != 0.f)
        {
            xt::svector<padding> paddings = {{0, 0}, padding_h, padding_w};

            ir::pad* pad_op = graph_.emplace<pad>(dt_float32, in_shape, paddings, pad_constant, pad_value);
            pad_op->name(op_name + ".pad(Convolution)");

            conv_op = graph_.emplace<conv2d>(pad_op->output().shape(), weight_shape, group, padding{0, 0}, padding{0, 0}, stride_h, stride_w, dilation_h, dilation_w, fused_activation);
            conv_op->name(op_name + ".conv2d(Convolution)");

            conv_op->input().connect(pad_op->output());
        }
        else
        {
            conv_op = graph_.emplace<conv2d>(in_shape, weight_shape, group, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, fused_activation);
            conv_op->name(op_name + ".conv2d(Convolution)");
        }

        ncnn::Mat weight_data = mb.load(weight_data_size, 0);
        ncnn::Mat bias_data;
        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
        }
        else
        {
            bias_data.create(num_output);
            bias_data.fill(0.f);
        }

        auto weight_node = graph_.emplace<constant>(dt_float32, weight_shape, std::span<const float>{ (float*)weight_data.data, (size_t)weight_data.w });
        conv_op->weights().connect(weight_node->output());

        auto bias_node = graph_.emplace<constant>(dt_float32, bias_shape, std::span<const float>{ (float*)bias_data.data, (size_t)bias_data.w });
        conv_op->bias().connect(bias_node->output());
    }

    ir::node* activation = 0;
    {
        if (activation_type == 2)
        {
            // leakyrelu
            const auto &alpha = graph_.emplace<constant>(activation_params[0]);
            alpha->name(op_name + ".alpha(LeakyReLU)");

            auto mul = graph_.emplace<binary>(binary_mul, conv_op->output().shape(), alpha->output().shape(), value_range<float>::full());
            mul->name(op_name + ".mul(LeakyReLU)");
            auto max = graph_.emplace<binary>(binary_max, conv_op->output().shape(), mul->output().shape(), value_range<float>::full());
            max->name(op_name + ".max(LeakyReLU)");

            mul->input_a().connect(conv_op->output());
            mul->input_b().connect(alpha->output());
            max->input_a().connect(conv_op->output());
            max->input_b().connect(mul->output());

            activation = max;
        }
        if (activation_type == 4)
        {
            // y = 1 / (1 + exp(-x))
            // y = exp(x) / (exp(x) + 1)
            auto exp = graph_.emplace<unary>(unary_exp, conv_op->output().shape());
            exp->name(op_name + ".exp(Sigmoid)");
            auto one = graph_.emplace<constant>(1.f);
            one->name(op_name + ".one(Sigmoid)");
            auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
            add->name(op_name + ".add(Sigmoid)");
            auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), add->output().shape(), value_range<float>::nonnegative());
            div->name(op_name + ".div(Sigmoid)");

            exp->input().connect(conv_op->output());
            add->input_a().connect(exp->output());
            add->input_b().connect(one->output());
            div->input_a().connect(exp->output());
            div->input_b().connect(add->output());

            activation = div;
        }
        if (activation_type == 5)
        {
            // y = x * tanh(log(exp(x) + 1))
            auto exp = graph_.emplace<unary>(unary_exp, conv_op->output().shape());
            exp->name(op_name + ".exp(Mish)");
            auto one = graph_.emplace<constant>(1.f);
            one->name(op_name + ".one(Mish)");
            auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
            add->name(op_name + ".add(Mish)");
            auto log = graph_.emplace<unary>(unary_log, add->output().shape());
            log->name(op_name + ".log(Mish)");
            auto tanh = graph_.emplace<unary>(unary_tanh, log->output().shape());
            tanh->name(op_name + ".tanh(Mish)");
            auto mul = graph_.emplace<binary>(binary_mul, conv_op->output().shape(), tanh->output().shape(), value_range<float>::nonnegative());
            mul->name(op_name + ".mul(Mish)");

            exp->input().connect(conv_op->output());
            add->input_a().connect(exp->output());
            add->input_b().connect(one->output());
            log->input().connect(add->output());
            tanh->input().connect(log->output());
            mul->input_a().connect(conv_op->output());
            mul->input_b().connect(tanh->output());

            activation = mul;
        }
    }

    if (activation)
        return activation;

    return conv_op;
}
