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
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_ReduceMax(const NodeProto &node)
{
    convert_reduce(node, reduce_max, std::numeric_limits<float>::lowest());
}

void onnx_importer::convert_op_ReduceMean(const NodeProto &node)
{
    convert_reduce(node, reduce_mean, 0.f);
}

void onnx_importer::convert_op_ReduceMin(const NodeProto &node)
{
    convert_reduce(node, reduce_min, std::numeric_limits<float>::max());
}

void onnx_importer::convert_op_ReduceSum(const NodeProto &node)
{
    convert_reduce(node, reduce_sum, 0.f);
}

void onnx_importer::convert_reduce(const NodeProto &node, const reduce_op_t reduce_op, const float init_value)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto &input_shape = get_shape(input);
    axis_t axes(input_shape.size());
    std::iota(begin(axes), end(axes), 0);

    const auto &axes_attr = get_attribute<axis_t>(node, "axes");
    if (axes_attr)
    {
        axes = axes_attr.value();
        std::transform(std::begin(axes), std::end(axes), std::begin(axes),
            [&input_shape](const auto e) { return real_axis(e, input_shape.size()); });
    }

    bool keepdims = true;
    const auto &keepdims_attr = get_attribute<int>(node, "keepdims");
    if (keepdims_attr)
        keepdims = static_cast<bool>(keepdims_attr.value());

    auto op = graph_.emplace<reduce>(reduce_op, input_shape, std::move(axes), init_value, keepdims);
    op->name(op_name + '(' + reduce_op_to_string(reduce_op) + ')');

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}

void onnx_importer::convert_op_ReduceL1(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
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

    // keepdims
    auto keepdims_attr = get_attribute<int>(node, "keepdims");
    bool keepdims = keepdims_attr ? keepdims_attr.value() == 1 : true;

    auto abs = graph_.emplace<unary>(unary_abs, input_shape);
    abs->name(op_name + ".abs(ReduceL1)");

    auto sum = graph_.emplace<reduce>(reduce_sum, abs->output().shape(), axes, 0.f, keepdims);
    sum->name(op_name + ".reduce_sum(ReduceL1)");

    sum->input().connect(abs->output());

    input_tensors_.emplace(&abs->input(), input);
    output_tensors_.emplace(output, &sum->output());
}

void onnx_importer::convert_op_ReduceL2(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
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

    // keepdims
    auto keepdims_attr = get_attribute<int>(node, "keepdims");
    bool keepdims = keepdims_attr ? keepdims_attr.value() == 1 : true;

    auto square = graph_.emplace<unary>(unary_square, input_shape);
    square->name(op_name + ".square(ReduceL2)");

    auto sum = graph_.emplace<reduce>(reduce_sum, square->output().shape(), axes, 0.f, keepdims);
    sum->name(op_name + ".reduce_sum(ReduceL2)");

    auto sqrt = graph_.emplace<unary>(unary_sqrt, sum->output().shape());
    sqrt->name(op_name + ".sqrt(ReduceL2)");

    sum->input().connect(square->output());
    sqrt->input().connect(sum->output());

    input_tensors_.emplace(&square->input(), input);
    output_tensors_.emplace(output, &sqrt->output());
}
