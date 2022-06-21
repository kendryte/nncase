/* Copyright 2019-2021 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ABS)
{
    convert_unary(op, unary_abs);
}

DEFINE_TFLITE_LOWER(CEIL)
{
    convert_unary(op, unary_ceil);
}

DEFINE_TFLITE_LOWER(COS)
{
    convert_unary(op, unary_cos);
}

DEFINE_TFLITE_LOWER(EXP)
{
    convert_unary(op, unary_exp);
}

DEFINE_TFLITE_LOWER(FLOOR)
{
    convert_unary(op, unary_floor);
}

DEFINE_TFLITE_LOWER(LOG)
{
    convert_unary(op, unary_log);
}

DEFINE_TFLITE_LOWER(NEG)
{
    convert_unary(op, unary_neg);
}

DEFINE_TFLITE_LOWER(ROUND)
{
    convert_unary(op, unary_round);
}

DEFINE_TFLITE_LOWER(RSQRT)
{
    convert_unary(op, unary_rsqrt);
}

DEFINE_TFLITE_LOWER(SIN)
{
    convert_unary(op, unary_sin);
}

DEFINE_TFLITE_LOWER(SQRT)
{
    convert_unary(op, unary_sqrt);
}

DEFINE_TFLITE_LOWER(SQUARE)
{
    convert_unary(op, unary_square);
}

DEFINE_TFLITE_LOWER(TANH)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto in_shape = get_shape(input.shape());
    auto input_type = to_data_type(input.type());

    auto two = graph_.emplace<constant>(2.f);
    auto mul = graph_.emplace<binary>(binary_mul, input_type, in_shape, two->output().shape(), value_range<float>::full());
    auto exp = graph_.emplace<unary>(unary_exp, mul->output().shape());
    auto one = graph_.emplace<constant>(1.f);
    auto sub = graph_.emplace<binary>(binary_sub, input_type, exp->output().shape(), one->output().shape(), value_range<float>::full());
    auto add = graph_.emplace<binary>(binary_add, input_type, exp->output().shape(), one->output().shape(), value_range<float>::full());
    auto div = graph_.emplace<binary>(binary_div, input_type, sub->output().shape(), add->output().shape(), value_range<float>::full());

    auto name = std::string(get_tensor(op.outputs(), 0).name()->string_view());
    two->name(name);
    mul->name(name);
    exp->name(name);
    one->name(name);
    sub->name(name);
    add->name(name);
    div->name(name);

    mul->input_b().connect(two->output());
    exp->input().connect(mul->output());
    sub->input_a().connect(exp->output());
    sub->input_b().connect(one->output());
    add->input_a().connect(exp->output());
    add->input_b().connect(one->output());
    div->input_a().connect(sub->output());
    div->input_b().connect(add->output());

    link_input_tensor(&mul->input_a(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &div->output());
}

void tflite_importer::convert_unary(const tflite::Operator &op, unary_op_t unary_op)
{
    auto &input = get_tensor(op.inputs(), 0);

    auto node = graph_.emplace<unary>(unary_op, get_shape(input.shape()));
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    link_input_tensor(&node->input(), op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), &node->output());
}
