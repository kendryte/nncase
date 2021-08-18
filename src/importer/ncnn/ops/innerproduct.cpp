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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_InnerProduct(const Layer &layer, const ParamDict &pd, const ModelBin &mb)
{
    const int num_output = pd.get(0, 0);
    const int bias_term = pd.get(1, 0);
    const int weight_data_size = pd.get(2, 0);
    const int activation_type = pd.get(9, 0);
    const Mat activation_params = pd.get(10, Mat());

    const int num_input = weight_data_size / num_output;

    const auto &op_name = layer.name;

    auto in_shape = layer.bottom_shapes[0];
    auto out_shape = layer.top_shapes[0];

    const shape_t in_shape_2rank = { 1, (size_t)num_input };
    const shape_t weight_shape_transposed = { (size_t)num_input, (size_t)num_output };
    const shape_t bias_shape = { (size_t)num_output };

    ir::bitcast *pre_reshape = graph_.emplace<bitcast>(dt_float32, in_shape, in_shape_2rank);
    pre_reshape->name(op_name + ".reshape_to_2d(InnerProduct)");

    input_tensors_.emplace(&pre_reshape->input(), layer.bottoms[0]);

    value_range<float> fused_activation = value_range<float>::full();
    {
        if (activation_type == 1)
        {
            fused_activation = { 0.f, INFINITY };
        }
        if (activation_type == 3)
        {
            fused_activation = { activation_params[0], activation_params[1] };
        }
    }

    ir::matmul *fc_op = 0;
    {
        fc_op = graph_.emplace<matmul>(in_shape_2rank, weight_shape_transposed, fused_activation);
        fc_op->name(op_name + ".matmul(InnerProduct)");

        fc_op->input_a().connect(pre_reshape->output());

        Mat weight_data = mb.load(weight_data_size, 0);
        Mat bias_data;
        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
        }
        else
        {
            bias_data.create(num_output);
            bias_data.fill(0.f);
        }

        // transpose weight
        ncnn::Mat weight_data_transposed(weight_data.w);
        {
            float *p0 = weight_data;
            float *p = weight_data_transposed;
            for (int i = 0; i < num_input; i++)
            {
                for (int j = 0; j < num_output; j++)
                {
                    p[i * num_output + j] = p0[j * num_input + i];
                }
            }
        }

        auto weight_node = graph_.emplace<constant>(dt_float32, weight_shape_transposed, std::span<const float> { (float *)weight_data_transposed.data, (size_t)weight_data_transposed.w });
        fc_op->input_b().connect(weight_node->output());

        auto bias_node = graph_.emplace<constant>(dt_float32, bias_shape, std::span<const float> { (float *)bias_data.data, (size_t)bias_data.w });
        fc_op->bias().connect(bias_node->output());
    }

    ir::bitcast *post_reshape = graph_.emplace<bitcast>(dt_float32, fc_op->output().shape(), out_shape);
    post_reshape->name(op_name + ".reshape_to_1d(InnerProduct)");

    post_reshape->input().connect(fc_op->output());

    if (activation_type == 0 || activation_type == 1 || activation_type == 3)
    {
        output_tensors_.emplace(layer.tops[0], &post_reshape->output());
    }
    if (activation_type == 2)
    {
        // leakyrelu
        const auto &alpha = graph_.emplace<constant>(activation_params[0]);
        alpha->name(op_name + ".alpha(LeakyReLU)");

        auto mul = graph_.emplace<binary>(binary_mul, post_reshape->output().shape(), alpha->output().shape(), value_range<float>::full());
        mul->name(op_name + ".mul(LeakyReLU)");
        auto max = graph_.emplace<binary>(binary_max, post_reshape->output().shape(), mul->output().shape(), value_range<float>::full());
        max->name(op_name + ".max(LeakyReLU)");

        mul->input_a().connect(post_reshape->output());
        mul->input_b().connect(alpha->output());
        max->input_a().connect(post_reshape->output());
        max->input_b().connect(mul->output());

        output_tensors_.emplace(layer.tops[0], &max->output());
    }
    if (activation_type == 4)
    {
        // y = 1 / (1 + exp(-x))
        // y = exp(x) / (exp(x) + 1)
        auto exp = graph_.emplace<unary>(unary_exp, post_reshape->output().shape());
        exp->name(op_name + ".exp(Sigmoid)");
        auto one = graph_.emplace<constant>(1.f);
        one->name(op_name + ".one(Sigmoid)");
        auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
        add->name(op_name + ".add(Sigmoid)");
        auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), add->output().shape(), value_range<float>::nonnegative());
        div->name(op_name + ".div(Sigmoid)");

        exp->input().connect(post_reshape->output());
        add->input_a().connect(exp->output());
        add->input_b().connect(one->output());
        div->input_a().connect(exp->output());
        div->input_b().connect(add->output());

        output_tensors_.emplace(layer.tops[0], &div->output());
    }
    if (activation_type == 5)
    {
        // y = x * tanh(log(exp(x) + 1))
        auto exp = graph_.emplace<unary>(unary_exp, post_reshape->output().shape());
        exp->name(op_name + ".exp(Mish)");
        auto one = graph_.emplace<constant>(1.f);
        one->name(op_name + ".one(Mish)");
        auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
        add->name(op_name + ".add(Mish)");
        auto log = graph_.emplace<unary>(unary_log, add->output().shape());
        log->name(op_name + ".log(Mish)");
        auto tanh = graph_.emplace<unary>(unary_tanh, log->output().shape());
        tanh->name(op_name + ".tanh(Mish)");
        auto mul = graph_.emplace<binary>(binary_mul, post_reshape->output().shape(), tanh->output().shape(), value_range<float>::nonnegative());
        mul->name(op_name + ".mul(Mish)");

        exp->input().connect(post_reshape->output());
        add->input_a().connect(exp->output());
        add->input_b().connect(one->output());
        log->input().connect(add->output());
        tanh->input().connect(log->output());
        mul->input_a().connect(post_reshape->output());
        mul->input_b().connect(tanh->output());

        output_tensors_.emplace(layer.tops[0], &mul->output());
    }
}
