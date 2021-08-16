// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

ir::node* nncase::importer::ncnn_importer::convert_op_ReLU(const ncnn::Layer &layer, const ncnn::ParamDict &pd, const ncnn::ModelBin& /*mb*/)
{
    float slope = pd.get(0, 0.f);

    const auto &op_name = layer.name;

    auto in_shape = layer.bottom_shapes[0];

    if (slope == 0.f)
    {
        auto zero = graph_.emplace<constant>(0.f);
        zero->name(op_name + ".zero(Relu)");
        auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());
        max->name(op_name + ".max(Relu)");

        max->input_b().connect(zero->output());

        return max;
    }
    else
    {
        const auto &alpha = graph_.emplace<constant>(slope);

        alpha->name(op_name + ".alpha(LeakyRelu)");

        auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
        mul->name(op_name + ".mul(LeakyRelu)");
        auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());
        max->name(op_name + ".max(LeakyRelu)");

        mul->input_b().connect(alpha->output());
        max->input_b().connect(mul->output());

        return max;
    }
}
