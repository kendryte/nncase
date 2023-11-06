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
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Add(const onnx::NodeProto &node)
{
    convert_binary(node, binary_add);
}

void onnx_importer::convert_op_Sub(const onnx::NodeProto &node)
{
    convert_binary(node, binary_sub);
}

void onnx_importer::convert_op_Mul(const onnx::NodeProto &node)
{
    convert_binary(node, binary_mul);
}

void onnx_importer::convert_op_Div(const onnx::NodeProto &node)
{
    convert_binary(node, binary_div);
}

void onnx_importer::convert_op_Min(const onnx::NodeProto &node)
{
    convert_binary(node, binary_min);
}

void onnx_importer::convert_op_Max(const onnx::NodeProto &node)
{
    convert_binary(node, binary_max);
}

void onnx_importer::convert_op_Pow(const onnx::NodeProto &node)
{
    convert_binary(node, binary_pow);
}

void onnx_importer::convert_binary(const onnx::NodeProto &node, const binary_op_t binary_op)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto &output = node.output()[0];

    auto input_a_shape = get_shape(input_a);
    const auto input_type = get_datatype(input_a).value();
    auto input_b_shape = get_shape(input_b);
    const auto input_b_type = get_datatype(input_b).value();
    convert *cvt = nullptr;
    if (input_type != input_b_type)
    {
        cvt = graph_.emplace<convert>(input_b_type, input_b_shape, input_type);
        cvt->name(op_name + "(Convert)");
    }
    auto op = graph_.emplace<binary>(binary_op, input_type, input_a_shape, input_b_shape, value_range<float>::full());
    op->name(op_name + '(' + binary_op_to_string(binary_op) + ')');

    input_tensors_.emplace(&op->input_a(), input_a);
    if (cvt)
    {
        input_tensors_.emplace(&cvt->input(), input_b);
        op->input_b().connect(cvt->output());
    }
    else
    {
        input_tensors_.emplace(&op->input_b(), input_b);
    }
    output_tensors_.emplace(output, &op->output());
}

void onnx_importer::convert_op_And(const onnx::NodeProto &node)
{
    convert_op_logical(node, binary_logical_and);
}

void onnx_importer::convert_op_logical(const onnx::NodeProto &node, const binary_op_t binary_op)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto &output = node.output()[0];

    quant_param_t qparam { 0, 1.f };
    auto deq_a = graph_.emplace<dequantize>(get_datatype(input_a).value(), get_shape(input_a), dt_float32, qparam);
    deq_a->name(op_name + "/deq_a");
    auto deq_b = graph_.emplace<dequantize>(get_datatype(input_b).value(), get_shape(input_b), dt_float32, qparam);
    deq_b->name(op_name + "/deq_b");

    auto op = graph_.emplace<binary>(binary_op, deq_a->output().type(), deq_a->output().shape(), deq_b->output().shape(), value_range<float>::full());
    op->name(op_name + '(' + binary_op_to_string(binary_op) + ')');
    op->input_a().connect(deq_a->output());
    op->input_b().connect(deq_b->output());

    auto quant = graph_.emplace<quantize>(dt_float32, op->output().shape(), get_datatype(output).value(), qparam);
    quant->name(op_name + "/quant");
    quant->input().connect(op->output());

    input_tensors_.emplace(&deq_a->input(), input_a);
    input_tensors_.emplace(&deq_b->input(), input_b);
    output_tensors_.emplace(output, &quant->output());
}
