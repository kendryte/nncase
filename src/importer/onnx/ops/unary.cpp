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
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Abs(const onnx::NodeProto &node)
{
    convert_unary(node, unary_abs);
}

void onnx_importer::convert_op_Acos(const onnx::NodeProto &node)
{
    convert_unary(node, unary_acos);
}

void onnx_importer::convert_op_Asin(const onnx::NodeProto &node)
{
    convert_unary(node, unary_asin);
}

void onnx_importer::convert_op_Ceil(const onnx::NodeProto &node)
{
    convert_unary(node, unary_ceil);
}

void onnx_importer::convert_op_Floor(const onnx::NodeProto &node)
{
    convert_unary(node, unary_floor);
}

void onnx_importer::convert_op_Cos(const onnx::NodeProto &node)
{
    convert_unary(node, unary_cos);
}

void onnx_importer::convert_op_Sign(const onnx::NodeProto &node)
{
    convert_unary(node, unary_sign);
}

void onnx_importer::convert_op_Sin(const onnx::NodeProto &node)
{
    convert_unary(node, unary_sin);
}

void onnx_importer::convert_op_Exp(const onnx::NodeProto &node)
{
    convert_unary(node, unary_exp);
}

void onnx_importer::convert_op_Log(const onnx::NodeProto &node)
{
    convert_unary(node, unary_log);
}

void onnx_importer::convert_op_Neg(const onnx::NodeProto &node)
{
    convert_unary(node, unary_neg);
}

void onnx_importer::convert_op_Not(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);
    const auto &op_name { generate_name(node) };

    // input
    const auto &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    assert(input_type == dt_uint8);
    auto input_shape = get_shape(input);

    // output
    const auto &output = node.output()[0];
    const auto output_type = get_datatype(output).value();
    assert(output_type == dt_uint8);

    // dequantize
    quant_param_t qparam { 0, 1.f };
    auto dequant = graph_.emplace<dequantize>(input_type, input_shape, dt_float32, qparam);
    dequant->name(op_name + "/dequant");

    // unary
    auto unary_op = unary_logical_not;
    auto op = graph_.emplace<unary>(unary_op, dequant->output().shape());
    op->name(op_name + '(' + unary_op_to_string(unary_op) + ')');
    op->input().connect(dequant->output());

    // quantize
    auto quant = graph_.emplace<quantize>(dt_float32, op->output().shape(), output_type, qparam);
    quant->name(op_name + "/quant");
    quant->input().connect(op->output());

    input_tensors_.emplace(&dequant->input(), input);
    output_tensors_.emplace(output, &quant->output());
}

void onnx_importer::convert_op_Round(const onnx::NodeProto &node)
{
    convert_unary(node, unary_round);
}

void onnx_importer::convert_op_Sqrt(const onnx::NodeProto &node)
{
    convert_unary(node, unary_sqrt);
}

void onnx_importer::convert_op_Rsqrt(const onnx::NodeProto &node)
{
    convert_unary(node, unary_rsqrt);
}

void onnx_importer::convert_op_Tanh(const onnx::NodeProto &node)
{
    convert_unary(node, unary_tanh);
}

void onnx_importer::convert_op_Erf(const onnx::NodeProto &node)
{
    convert_unary(node, unary_erf);
}

void onnx_importer::convert_unary(const onnx::NodeProto &node, const unary_op_t unary_op)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto &input_shape = get_shape(input);
    auto op = graph_.emplace<unary>(unary_op, input_shape);
    op->name(op_name + '(' + unary_op_to_string(unary_op) + ')');

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}

// Sinh(x) = (exp(x) - exp(-x)) / 2
void onnx_importer::convert_op_Sinh(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto &in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    auto exp1 = graph_.emplace<unary>(unary_exp, in_shape);
    exp1->name(op_name + ".exp(Sinh)");

    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    neg->name(op_name + ".neg(Sinh)");

    auto exp2 = graph_.emplace<unary>(unary_exp, neg->output().shape());
    exp2->name(op_name + ".exp(Sinh)");

    auto sub = graph_.emplace<binary>(binary_sub, input_type, exp1->output().shape(), exp2->output().shape(), value_range<float>::full());
    sub->name(op_name + ".sub(Sinh)");

    auto two = graph_.emplace<constant>(2.f);
    two->name(op_name + ".two(Sinh)");

    auto div = graph_.emplace<binary>(binary_div, input_type, sub->output().shape(), two->output().shape(), value_range<float>::full());
    div->name(op_name + ".div(Sinh)");

    exp2->input().connect(neg->output());
    sub->input_a().connect(exp1->output());
    sub->input_b().connect(exp2->output());
    div->input_a().connect(sub->output());
    div->input_b().connect(two->output());

    input_tensors_.emplace(&exp1->input(), input);
    input_tensors_.emplace(&neg->input(), input);
    output_tensors_.emplace(output, &div->output());
}

// Cosh(x) = (exp(x) + exp(-x)) / 2
void onnx_importer::convert_op_Cosh(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto &in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    auto exp1 = graph_.emplace<unary>(unary_exp, in_shape);
    exp1->name(op_name + ".exp(Cosh)");

    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    neg->name(op_name + ".neg(Cosh)");

    auto exp2 = graph_.emplace<unary>(unary_exp, neg->output().shape());
    exp2->name(op_name + ".exp(Cosh)");

    auto add = graph_.emplace<binary>(binary_add, input_type, exp1->output().shape(), exp2->output().shape(), value_range<float>::nonnegative());
    add->name(op_name + ".add(Cosh)");

    auto two = graph_.emplace<constant>(2.f);
    two->name(op_name + ".two(Cosh)");

    auto div = graph_.emplace<binary>(binary_div, input_type, add->output().shape(), two->output().shape(), value_range<float>::nonnegative());
    div->name(op_name + ".div(Cosh)");

    exp2->input().connect(neg->output());
    add->input_a().connect(exp1->output());
    add->input_b().connect(exp2->output());
    div->input_a().connect(add->output());
    div->input_b().connect(two->output());

    input_tensors_.emplace(&exp1->input(), input);
    input_tensors_.emplace(&neg->input(), input);
    output_tensors_.emplace(output, &div->output());
}

// Asinh(x) = ln(x + sqrt(x^2 + 1))
void onnx_importer::convert_op_Asinh(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto &in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    struct value_range<float> one_to_infinity =
    {
        1, std::numeric_limits<float>::max()
    };

    auto square = graph_.emplace<unary>(unary_square, in_shape);
    square->name(op_name + ".square(Asinh)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Asinh)");

    auto add1 = graph_.emplace<binary>(binary_add, input_type, square->output().shape(), one->output().shape(), one_to_infinity);
    add1->name(op_name + ".add1(Asinh)");

    auto sqrt = graph_.emplace<unary>(unary_sqrt, add1->output().shape());
    sqrt->name(op_name + ".sqrt(Asinh)");

    auto add2 = graph_.emplace<binary>(binary_add, input_type, in_shape, sqrt->output().shape(), one_to_infinity);
    add2->name(op_name + ".add2(Asinh)");

    auto log = graph_.emplace<unary>(unary_log, add2->output().shape());
    log->name(op_name + ".log(Asinh)");

    add1->input_a().connect(square->output());
    add1->input_b().connect(one->output());
    sqrt->input().connect(add1->output());
    add2->input_b().connect(sqrt->output());
    log->input().connect(add2->output());

    input_tensors_.emplace(&square->input(), input);
    input_tensors_.emplace(&add2->input_a(), input);
    output_tensors_.emplace(output, &log->output());
}

// Acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1
void onnx_importer::convert_op_Acosh(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto &in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    struct value_range<float> one_to_infinity =
    {
        1, std::numeric_limits<float>::max()
    };

    auto square = graph_.emplace<unary>(unary_square, in_shape);
    square->name(op_name + ".square(Acosh)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Acosh)");

    auto sub = graph_.emplace<binary>(binary_sub, input_type, square->output().shape(), one->output().shape(), value_range<float>::nonnegative());
    sub->name(op_name + ".sub(Acosh)");

    auto sqrt = graph_.emplace<unary>(unary_sqrt, sub->output().shape());
    sqrt->name(op_name + ".sqrt(Acosh)");

    auto add = graph_.emplace<binary>(binary_add, input_type, in_shape, sqrt->output().shape(), one_to_infinity);
    add->name(op_name + ".add(Acosh)");

    auto log = graph_.emplace<unary>(unary_log, add->output().shape());
    log->name(op_name + ".log(Acosh)");

    sub->input_a().connect(square->output());
    sub->input_b().connect(one->output());
    sqrt->input().connect(sub->output());
    add->input_b().connect(sqrt->output());
    log->input().connect(add->output());

    input_tensors_.emplace(&square->input(), input);
    input_tensors_.emplace(&add->input_a(), input);
    output_tensors_.emplace(output, &log->output());
}
