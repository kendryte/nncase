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
#include <algorithm>
#include <cassert>
#include <limits>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_MeanVarianceNormalization(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    // axes
    axis_t axes(input_shape.size());
    std::iota(begin(axes), end(axes), 0);
    const auto &axes_attr = get_attribute<axis_t>(node, "axes");
    if (axes_attr)
    {
        axes = axes_attr.value();
        std::transform(std::begin(axes), std::end(axes), std::begin(axes),
            [&input_shape](const auto e) { return real_axis(e, input_shape.size()); });
    }

    auto mean1 = graph_.emplace<reduce>(reduce_mean, input_type, input_shape, axes, 0.f, true);
    mean1->name(op_name + ".reduce_mean1(MeanVarianceNormalization)");

    auto sub1 = graph_.emplace<binary>(binary_sub, input_type, input_shape, mean1->output().shape(), value_range<float>::full());
    sub1->name(op_name + ".reduce_sub1(MeanVarianceNormalization)");
    sub1->input_b().connect(mean1->output());

    auto square1 = graph_.emplace<unary>(unary_square, input_shape);
    square1->name(op_name + ".reduce_square1(MeanVarianceNormalization)");

    auto square2 = graph_.emplace<unary>(unary_square, mean1->output().shape());
    square2->name(op_name + ".reduce_square2(MeanVarianceNormalization)");
    square2->input().connect(mean1->output());

    auto mean2 = graph_.emplace<reduce>(reduce_mean, input_type, square1->output().shape(), axes, 0.f, true);
    mean2->name(op_name + ".reduce_mean2(MeanVarianceNormalization)");
    mean2->input().connect(square1->output());

    auto sub2 = graph_.emplace<binary>(binary_sub, input_type, mean2->output().shape(), square2->output().shape(), value_range<float>::full());
    sub2->name(op_name + ".reduce_sub2(MeanVarianceNormalization)");
    sub2->input_a().connect(mean2->output());
    sub2->input_b().connect(square2->output());

    auto sqrt = graph_.emplace<unary>(unary_sqrt, sub2->output().shape());
    sqrt->name(op_name + ".reduce_sqrt(MeanVarianceNormalization)");
    sqrt->input().connect(sub2->output());

    auto div = graph_.emplace<binary>(binary_div, input_type, sub1->output().shape(), sqrt->output().shape(), value_range<float>::full());
    div->name(op_name + ".reduce_div(MeanVarianceNormalization)");
    div->input_a().connect(sub1->output());
    div->input_b().connect(sqrt->output());

    input_tensors_.emplace(&mean1->input(), input);
    input_tensors_.emplace(&sub1->input_a(), input);
    input_tensors_.emplace(&square1->input(), input);
    output_tensors_.emplace(output, &div->output());
}
