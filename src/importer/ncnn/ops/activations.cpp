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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_Clip(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const float min = pd.get(0, -FLT_MAX);
    const float max = pd.get(1, FLT_MAX);

    const auto &op_name = layer.name;

    auto in_shape = output_tensors_.at(layer.bottoms[0])->shape();

    auto min_op = graph_.emplace<constant>(min);
    min_op->name(op_name + ".min(Clip)");

    auto max_op = graph_.emplace<constant>(max);
    max_op->name(op_name + ".max(Clip)");

    auto clamp_op = graph_.emplace<clamp>(in_shape, min_op->output().shape(), max_op->output().shape());
    clamp_op->name(op_name + ".clamp(Clip)");

    clamp_op->input_low().connect(min_op->output());
    clamp_op->input_high().connect(max_op->output());

    input_tensors_.emplace(&clamp_op->input(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &clamp_op->output());
}

void nncase::importer::ncnn_importer::convert_op_ReLU(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const float slope = pd.get(0, 0.f);

    const auto &op_name = layer.name;

    auto in_shape = output_tensors_.at(layer.bottoms[0])->shape();

    if (slope == 0.f)
    {
        auto zero = graph_.emplace<constant>(0.f);
        zero->name(op_name + ".zero(ReLU)");
        auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());
        max->name(op_name + ".max(ReLU)");

        max->input_b().connect(zero->output());

        input_tensors_.emplace(&max->input_a(), layer.bottoms[0]);
        output_tensors_.emplace(layer.tops[0], &max->output());
    }
    else
    {
        const auto &alpha = graph_.emplace<constant>(slope);

        alpha->name(op_name + ".alpha(LeakyReLU)");

        auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
        mul->name(op_name + ".mul(LeakyReLU)");
        auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());
        max->name(op_name + ".max(LeakyReLU)");

        mul->input_b().connect(alpha->output());
        max->input_b().connect(mul->output());

        input_tensors_.emplace(&mul->input_a(), layer.bottoms[0]);
        input_tensors_.emplace(&max->input_a(), layer.bottoms[0]);
        output_tensors_.emplace(layer.tops[0], &max->output());
    }
}

void nncase::importer::ncnn_importer::convert_op_Sigmoid(const Layer &layer, const ParamDict & /*pd*/, const ModelBin & /*mb*/)
{
    const auto &op_name = layer.name;

    auto in_shape = output_tensors_.at(layer.bottoms[0])->shape();

    // y = exp(x) / (exp(x) + 1)
    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Sigmoid)");
    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Sigmoid)");
    auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
    add->name(op_name + ".add(Sigmoid)");
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), add->output().shape(), value_range<float>::nonnegative());
    div->name(op_name + ".div(Sigmoid)");
    add->input_a().connect(exp->output());
    add->input_b().connect(one->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&exp->input(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &div->output());
}

void nncase::importer::ncnn_importer::convert_op_HardSigmoid(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const float alpha = pd.get(0, 0.2f);
    const float beta = pd.get(1, 0.5f);

    const auto &op_name = layer.name;

    auto in_shape = output_tensors_.at(layer.bottoms[0])->shape();

    // y = max(0, min(1, alpha * x + beta))
    const auto &alpha_constant = graph_.emplace<constant>(alpha);
    alpha_constant->name(op_name + ".alpha(HardSigmoid)");

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha_constant->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(HardSigmoid)");

    const auto &beta_constant = graph_.emplace<constant>(beta);
    beta_constant->name(op_name + ".beta(HardSigmoid)");

    auto add = graph_.emplace<binary>(binary_add, mul->output().shape(), beta_constant->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(HardSigmoid)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSigmoid)");
    auto min = graph_.emplace<binary>(binary_min, add->output().shape(), one->output().shape(), value_range<float>::full());

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSigmoid)");
    auto max = graph_.emplace<binary>(binary_max, min->output().shape(), zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(HardSigmoid)");

    mul->input_b().connect(alpha_constant->output());
    add->input_a().connect(mul->output());
    add->input_b().connect(beta_constant->output());
    min->input_a().connect(add->output());
    min->input_b().connect(one->output());
    max->input_a().connect(min->output());
    max->input_b().connect(zero->output());

    input_tensors_.emplace(&mul->input_a(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &max->output());
}

void nncase::importer::ncnn_importer::convert_op_HardSwish(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const float alpha = pd.get(0, 0.2f);
    const float beta = pd.get(1, 0.5f);

    const auto &op_name = layer.name;

    auto in_shape = output_tensors_.at(layer.bottoms[0])->shape();

    // y = x * max(0, min(1, alpha * x + beta))
    const auto &alpha_constant = graph_.emplace<constant>(alpha);
    alpha_constant->name(op_name + ".alpha(HardSwish)");

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha_constant->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(HardSwish)");

    const auto &beta_constant = graph_.emplace<constant>(beta);
    beta_constant->name(op_name + ".beta(HardSwish)");

    auto add = graph_.emplace<binary>(binary_add, mul->output().shape(), beta_constant->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(HardSwish)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSwish)");
    auto min = graph_.emplace<binary>(binary_min, add->output().shape(), one->output().shape(), value_range<float>::full());

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSwish)");
    auto max = graph_.emplace<binary>(binary_max, min->output().shape(), zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(HardSwish)");

    auto mul2 = graph_.emplace<binary>(binary_mul, in_shape, max->output().shape(), value_range<float>::full());
    mul2->name(op_name + ".mul2(HardSwish)");

    mul->input_b().connect(alpha_constant->output());
    add->input_a().connect(mul->output());
    add->input_b().connect(beta_constant->output());
    min->input_a().connect(add->output());
    min->input_b().connect(one->output());
    max->input_a().connect(min->output());
    max->input_b().connect(zero->output());
    mul2->input_b().connect(max->output());

    input_tensors_.emplace(&mul->input_a(), layer.bottoms[0]);
    input_tensors_.emplace(&mul2->input_a(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &mul2->output());
}
