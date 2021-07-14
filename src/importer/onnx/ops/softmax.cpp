/* Copyright 2020 Alexey Chernov <4ernov@gmail.com>
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

#include "../onnx_importer.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Softmax(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);

    auto axis_attr = get_attribute<int64_t>(node, "axis");
    int axis = !axis_attr ? 1 : static_cast<int>(axis_attr.value());
    axis_t reduce_axis { static_cast<int>(real_axis(axis, input_shape.size())) };

    auto max = graph_.emplace<reduce>(reduce_max, input_shape, reduce_axis, std::numeric_limits<float>::lowest(), true);
    max->name(op_name + ".max(Softmax)");
    auto sub = graph_.emplace<binary>(binary_sub, input_shape, max->output().shape(), value_range<float>::full());
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

    input_tensors_.emplace(&max->input(), input);
    input_tensors_.emplace(&sub->input_a(), input);
    output_tensors_.emplace(output, &div->output());
}
