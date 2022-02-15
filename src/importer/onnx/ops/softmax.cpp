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
#include <nncase/ir/ops/bitcast.h>
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
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);

    auto opset_version = get_opset_version();
    int64_t default_axis = opset_version >= 13 ? -1 : 1;
    auto axis_value = static_cast<int>(get_attribute<int64_t>(node, "axis").value_or(default_axis));
    auto axis = static_cast<int>(real_axis(axis_value, input_shape.size()));

    // opset 1/11
    // 1. The input should be reshaped as 2d tenor to compute Softmax.
    // 2. The output should be reshaped as original shape of input.
    if (opset_version < 13)
    {
        shape_t new_shape;
        size_t dim = 1;
        for (auto i = 0; i < axis; i++)
            dim *= input_shape[i];
        new_shape.push_back(dim);

        dim = 1;
        for (auto i = axis; i < input_shape.size(); i++)
            dim *= input_shape[i];
        new_shape.push_back(dim);

        auto bc1 = graph_.emplace<bitcast>(input_type, input_shape, new_shape);
        bc1->name(op_name + ".bitcast1(Softmax)");

        axis_t axes { 1 };
        auto rmax = graph_.emplace<reduce>(reduce_max, bc1->output().shape(), axes, std::numeric_limits<float>::lowest(), true);
        rmax->name(op_name + ".rmax(Softmax)");

        auto sub = graph_.emplace<binary>(binary_sub, bc1->output().shape(), rmax->output().shape(), value_range<float>::full());
        sub->name(op_name + ".sub(Softmax)");

        auto exp = graph_.emplace<unary>(unary_exp, sub->output().shape());
        exp->name(op_name + ".exp(Softmax)");

        auto rsum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), axes, 0.f, true);
        rsum->name(op_name + ".rsum(Softmax)");

        auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), rsum->output().shape(), value_range<float>::full());
        div->name(op_name + ".div(Softmax)");

        auto bc2 = graph_.emplace<bitcast>(input_type, div->output().shape(), input_shape);
        bc2->name(op_name + ".bitcast2(Softmax)");

        rmax->input().connect(bc1->output());
        sub->input_a().connect(bc1->output());
        sub->input_b().connect(rmax->output());
        exp->input().connect(sub->output());
        rsum->input().connect(exp->output());
        div->input_a().connect(exp->output());
        div->input_b().connect(rsum->output());
        bc2->input().connect(div->output());

        input_tensors_.emplace(&bc1->input(), input);
        output_tensors_.emplace(output, &bc2->output());
    }
    else
    {
        axis_t axes { axis };
        auto rmax = graph_.emplace<reduce>(reduce_max, input_shape, axes, std::numeric_limits<float>::lowest(), true);
        rmax->name(op_name + ".rmax(Softmax)");

        auto sub = graph_.emplace<binary>(binary_sub, input_shape, rmax->output().shape(), value_range<float>::full());
        sub->name(op_name + ".sub(Softmax)");

        auto exp = graph_.emplace<unary>(unary_exp, sub->output().shape());
        exp->name(op_name + ".exp(Softmax)");

        auto rsum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), axes, 0.f, true);
        rsum->name(op_name + ".rsum(Softmax)");

        auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), rsum->output().shape(), value_range<float>::full());
        div->name(op_name + ".div(Softmax)");

        sub->input_b().connect(rmax->output());
        exp->input().connect(sub->output());
        rsum->input().connect(exp->output());
        div->input_a().connect(exp->output());
        div->input_b().connect(rsum->output());

        input_tensors_.emplace(&rmax->input(), input);
        input_tensors_.emplace(&sub->input_a(), input);
        output_tensors_.emplace(output, &div->output());
    }
}

// LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
void onnx_importer::convert_op_LogSoftmax(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);

    auto opset_version = get_opset_version();
    int64_t default_axis = opset_version >= 13 ? -1 : 1;
    auto axis_value = static_cast<int>(get_attribute<int64_t>(node, "axis").value_or(default_axis));
    auto axis = static_cast<int>(real_axis(axis_value, input_shape.size()));

    // opset 1/11
    // 1. The input should be reshaped as 2d tenor to compute LogSoftmax.
    // 2. The output should be reshaped as original shape of input.
    if (opset_version < 13)
    {
        shape_t new_shape;
        size_t dim = 1;
        for (auto i = 0; i < axis; i++)
            dim *= input_shape[i];
        new_shape.push_back(dim);

        dim = 1;
        for (auto i = axis; i < input_shape.size(); i++)
            dim *= input_shape[i];
        new_shape.push_back(dim);

        auto bc1 = graph_.emplace<bitcast>(input_type, input_shape, new_shape);
        bc1->name(op_name + ".bitcast1(LogSoftmax)");

        axis_t axes { 1 };
        auto rmax = graph_.emplace<reduce>(reduce_max, bc1->output().shape(), axes, std::numeric_limits<float>::lowest(), true);
        rmax->name(op_name + ".rmax(LogSoftmax)");

        auto sub1 = graph_.emplace<binary>(binary_sub, bc1->output().shape(), rmax->output().shape(), value_range<float>::full());
        sub1->name(op_name + ".sub1(LogSoftmax)");

        auto exp = graph_.emplace<unary>(unary_exp, sub1->output().shape());
        exp->name(op_name + ".exp(LogSoftmax)");

        auto rsum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), axes, 0.f, true);
        rsum->name(op_name + ".rsum(LogSoftmax)");

        auto log = graph_.emplace<unary>(unary_log, rsum->output().shape());
        log->name(op_name + ".log(LogSoftmax)");

        auto sub2 = graph_.emplace<binary>(binary_sub, sub1->output().shape(), log->output().shape(), value_range<float>::full());
        sub2->name(op_name + ".sub2(LogSoftmax)");

        auto bc2 = graph_.emplace<bitcast>(input_type, sub2->output().shape(), input_shape);
        bc2->name(op_name + ".bitcast2(LogSoftmax)");

        rmax->input().connect(bc1->output());
        sub1->input_a().connect(bc1->output());
        sub1->input_b().connect(rmax->output());
        exp->input().connect(sub1->output());
        rsum->input().connect(exp->output());
        log->input().connect(rsum->output());
        sub2->input_a().connect(sub1->output());
        sub2->input_b().connect(log->output());
        bc2->input().connect(sub2->output());

        input_tensors_.emplace(&bc1->input(), input);
        output_tensors_.emplace(output, &bc2->output());
    }
    else
    {
        // opset 13
        axis_t axes { axis };
        auto rmax = graph_.emplace<reduce>(reduce_max, input_shape, axes, std::numeric_limits<float>::lowest(), true);
        rmax->name(op_name + ".rmax(LogSoftmax)");

        auto sub1 = graph_.emplace<binary>(binary_sub, input_shape, rmax->output().shape(), value_range<float>::full());
        sub1->name(op_name + ".sub1(LogSoftmax)");

        auto exp = graph_.emplace<unary>(unary_exp, sub1->output().shape());
        exp->name(op_name + ".exp(LogSoftmax)");

        auto rsum = graph_.emplace<reduce>(reduce_sum, exp->output().shape(), axes, 0.f, true);
        rsum->name(op_name + ".rsum(LogSoftmax)");

        auto log = graph_.emplace<unary>(unary_log, rsum->output().shape());
        log->name(op_name + ".log(LogSoftmax)");

        auto sub2 = graph_.emplace<binary>(binary_sub, sub1->output().shape(), log->output().shape(), value_range<float>::full());
        sub2->name(op_name + ".sub2(LogSoftmax)");

        sub1->input_b().connect(rmax->output());
        exp->input().connect(sub1->output());
        rsum->input().connect(exp->output());
        log->input().connect(rsum->output());
        sub2->input_a().connect(sub1->output());
        sub2->input_b().connect(log->output());

        input_tensors_.emplace(&rmax->input(), input);
        input_tensors_.emplace(&sub1->input_a(), input);
        output_tensors_.emplace(output, &sub2->output());
    }
}