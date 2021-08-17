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
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_Softmax(const Layer &layer, const ParamDict &pd, const ModelBin& /*mb*/)
{
    const int axis = pd.get(0, 0);

    const auto &op_name = layer.name;

    auto in_shape = layer.bottom_shapes[0];

    axis_t reduce_axis = {axis};

    auto max = graph_.emplace<reduce>(reduce_max, in_shape, reduce_axis, std::numeric_limits<float>::lowest(), true);
    max->name(op_name + ".max(Softmax)");
    auto sub = graph_.emplace<binary>(binary_sub, in_shape, max->output().shape(), value_range<float>::full());
    sub->name(op_name + ".sub(Softmax)");
    auto exp = graph_.emplace<unary>(unary_exp, sub->output().shape());
    exp->name(op_name + ".exp(Softmax)");
    auto sum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), reduce_axis, 0.f, true);
    sum->name(op_name + ".sum(Softmax)");
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), sum->output().shape(), value_range<float>::full());
    div->name(op_name + ".div(Softmax)");

    sub->input_b().connect(max->output());
    exp->input().connect(sub->output());
    sum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(sum->output());

    input_tensors_.emplace(&max->input(), layer.bottoms[0]);
    input_tensors_.emplace(&sub->input_a(), layer.bottoms[0]);
    output_tensors_.emplace(layer.tops[0], &div->output());
}
