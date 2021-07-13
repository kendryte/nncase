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

void onnx_importer::convert_op_LpNormalization(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto &input_shape = get_shape(input);

    axis_t reduce_axis { static_cast<int>(real_axis(get_attribute<int>(node, "axis").value(), input_shape.size())) };
    const auto p = get_attribute<int>(node, "p").value();

    assert(p >= 1 && p <= 2);

    switch (p)
    {
    case 1:
    {
        auto abs = graph_.emplace<unary>(unary_abs, input_shape);
        abs->name(op_name + ".abs(L1Normalization)");
        auto sum = graph_.emplace<reduce>(reduce_sum, abs->output().shape(), reduce_axis, 0.f, true);
        sum->name(op_name + ".reduce_sum(L1Normalization)");
        auto div = graph_.emplace<binary>(binary_div, input_shape, sum->output().shape(), value_range<float>::full());
        div->name(op_name + ".div(L1Normalization)");

        sum->input().connect(abs->output());
        div->input_b().connect(sum->output());

        input_tensors_.emplace(&abs->input(), input);
        input_tensors_.emplace(&div->input_a(), input);
        output_tensors_.emplace(output, &div->output());
        break;
    }
    case 2:
    {
        auto square = graph_.emplace<unary>(unary_square, input_shape);
        square->name(op_name + ".square(L2Normalization)");
        auto sum = graph_.emplace<reduce>(reduce_sum, square->output().shape(), reduce_axis, 0.f, true);
        sum->name(op_name + ".reduce_sum(L2Normalization)");
        auto epsilon = graph_.emplace<constant>(1e-10f);
        epsilon->name(op_name + ".eps(L2Normalization)");
        auto max = graph_.emplace<binary>(binary_max, sum->output().shape(), epsilon->output().shape(), value_range<float>::full());
        max->name(op_name + ".stab(L2Normalization)");
        auto sqrt = graph_.emplace<unary>(unary_sqrt, max->output().shape());
        sqrt->name(op_name + ".sqrt(L2Normalization)");
        auto div = graph_.emplace<binary>(binary_div, input_shape, sqrt->output().shape(), value_range<float>::full());
        div->name(op_name + ".div(L2Normalization)");

        sum->input().connect(square->output());
        max->input_a().connect(sum->output());
        max->input_b().connect(epsilon->output());
        sqrt->input().connect(max->output());
        div->input_b().connect(sqrt->output());

        input_tensors_.emplace(&square->input(), input);
        input_tensors_.emplace(&div->input_a(), input);
        output_tensors_.emplace(output, &div->output());
        break;
    }
    default:
    {
        break;
    }
    }
}
