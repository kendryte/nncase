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
#include <nncase/ir/ops/unary.h>

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
    auto op = graph_.emplace<binary>(binary_op, input_type, input_a_shape, input_b_shape, value_range<float>::full());
    op->name(op_name + '(' + binary_op_to_string(binary_op) + ')');

    input_tensors_.emplace(&op->input_a(), input_a);
    input_tensors_.emplace(&op->input_b(), input_b);
    output_tensors_.emplace(output, &op->output());
}

void onnx_importer::convert_op_And(const onnx::NodeProto &node)
{
    convert_op_logical(node, binary_logical_and);
}

void onnx_importer::convert_op_Or(const onnx::NodeProto &node)
{
    convert_op_logical(node, binary_logical_or);
}

void onnx_importer::convert_op_Xor(const onnx::NodeProto &node)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto &output = node.output()[0];
    auto input_a_shape = get_shape(input_a);
    auto input_b_shape = get_shape(input_b);

    quant_param_t qparam { 0, 1.f };
    auto deq_a = graph_.emplace<dequantize>(get_datatype(input_a).value(), get_shape(input_a), dt_float32, qparam);
    deq_a->name(op_name + "/deq_a");
    auto deq_b = graph_.emplace<dequantize>(get_datatype(input_b).value(), get_shape(input_b), dt_float32, qparam);
    deq_b->name(op_name + "/deq_b");

    auto not1 = graph_.emplace<unary>(unary_logical_not, deq_a->output().shape());
    not1->name(op_name + '(' + unary_op_to_string(unary_logical_not) + ')' + '1');
    not1->input().connect(deq_a->output());
    auto not2 = graph_.emplace<unary>(unary_logical_not, deq_b->output().shape());
    not2->name(op_name + '(' + unary_op_to_string(unary_logical_not) + ')' + '2');
    not2->input().connect(deq_b->output());
    auto and1 = graph_.emplace<binary>(binary_logical_and, not1->output().type(), not1->output().shape(), input_b_shape, value_range<float>::full());
    and1->name(op_name + '(' + binary_op_to_string(binary_logical_and) + ')' + '1');
    auto and2 = graph_.emplace<binary>(binary_logical_and, not2->output().type(), not2->output().shape(), input_a_shape, value_range<float>::full());
    and2->name(op_name + '(' + binary_op_to_string(binary_logical_and) + ')' + '2');
    auto or1 = graph_.emplace<binary>(binary_logical_or, and1->output().type(), and1->output().shape(), and2->output().shape(), value_range<float>::full());
    or1->name(op_name + '(' + binary_op_to_string(binary_logical_or) + ')');

    not1->input().connect(deq_a->output());
    not2->input().connect(deq_b->output());
    and1->input_a().connect(not1->output());
    and2->input_a().connect(not2->output());
    and1->input_b().connect(deq_b->output());
    and2->input_b().connect(deq_a->output());
    or1->input_a().connect(and1->output());
    or1->input_b().connect(and2->output());

    auto quant = graph_.emplace<quantize>(dt_float32, or1->output().shape(), get_datatype(output).value(), qparam);
    quant->name(op_name + "/quant");
    quant->input().connect(or1->output());

    input_tensors_.emplace(&deq_a->input(), input_a);
    input_tensors_.emplace(&deq_b->input(), input_b);
    output_tensors_.emplace(output, &quant->output());
}

void onnx_importer::convert_op_Mod(const onnx::NodeProto &node)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto &output = node.output()[0];
    const auto input_type = get_datatype(input_a).value();
    auto input_a_shape = get_shape(input_a);
    auto input_b_shape = get_shape(input_b);

    // input_type is float
    if (input_type == 9)
    {
        auto div = graph_.emplace<binary>(binary_div, input_type, input_a_shape, input_b_shape, value_range<float>::full());
        div->name(op_name + '(' + binary_op_to_string(binary_div) + ')');

        auto floor = graph_.emplace<unary>(unary_floor, div->output().shape());
        floor->name(op_name + '(' + unary_op_to_string(unary_floor) + ')');

        auto mul = graph_.emplace<binary>(binary_mul, floor->output().type(), floor->output().shape(), input_b_shape, value_range<float>::full());
        mul->name(op_name + '(' + binary_op_to_string(binary_mul) + ')');

        auto sub = graph_.emplace<binary>(binary_sub, input_type, input_a_shape, mul->output().shape(), value_range<float>::full());
        sub->name(op_name + '(' + binary_op_to_string(binary_sub) + ')');

        floor->input().connect(div->output());
        mul->input_a().connect(floor->output());
        sub->input_b().connect(mul->output());

        input_tensors_.emplace(&div->input_a(), input_a);
        input_tensors_.emplace(&sub->input_a(), input_a);
        input_tensors_.emplace(&div->input_b(), input_b);
        input_tensors_.emplace(&mul->input_b(), input_b);
        output_tensors_.emplace(output, &sub->output());
    }
    else
    {
        auto div = graph_.emplace<binary>(binary_div, input_type, input_a_shape, input_b_shape, value_range<float>::full());
        div->name(op_name + '(' + binary_op_to_string(binary_div) + ')');

        auto mul = graph_.emplace<binary>(binary_mul, div->output().type(), div->output().shape(), input_b_shape, value_range<float>::full());
        mul->name(op_name + '(' + binary_op_to_string(binary_mul) + ')');

        auto sub = graph_.emplace<binary>(binary_sub, input_type, input_a_shape, mul->output().shape(), value_range<float>::full());
        sub->name(op_name + '(' + binary_op_to_string(binary_sub) + ')');

        mul->input_a().connect(div->output());
        sub->input_b().connect(mul->output());

        input_tensors_.emplace(&div->input_a(), input_a);
        input_tensors_.emplace(&sub->input_a(), input_a);
        input_tensors_.emplace(&div->input_b(), input_b);
        input_tensors_.emplace(&mul->input_b(), input_b);
        output_tensors_.emplace(output, &sub->output());
    }
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
