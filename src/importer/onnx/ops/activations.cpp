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
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/compare.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/trilu.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Relu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Relu)");
    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(generate_name(node) + ".max(Relu)");

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_LeakyRelu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto &&in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(0.01f);
    const auto &alpha = graph_.emplace<constant>(alpha_value);

    alpha->name(op_name + ".alpha(LeakyRelu)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, in_shape, alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(LeakyRelu)");
    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, mul->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(LeakyRelu)");

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&mul->input_a(), input);
    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_PRelu(const NodeProto &node)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &slope = node.input()[1];
    const auto &output = node.output()[0];

    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    auto slope_shape = get_shape(slope);

    constant *alpha = nullptr;
    auto init = get_initializer(slope);
    if (init)
    {
        // slope is initializer
        auto slope_value = to<std::vector<float>>(init.value());
        alpha = graph_.emplace<constant>(get_datatype<float>(), slope_shape, slope_value);
        alpha->name(op_name + ".alpha(PRelu)");
    }
    else
    {
        // slope is constant node
        auto it = output_tensors_.find(slope);
        if (it != output_tensors_.end())
        {
            alpha = dynamic_cast<constant *>(&it->second->owner());
        }
    }
    assert(alpha != nullptr);

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(PRelu)");

    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(PRelu)");

    auto min = graph_.emplace<binary>(binary_min, input_type, in_shape, zero->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(PRelu)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, min->output().shape(), alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(PRelu)");

    auto add = graph_.emplace<binary>(binary_add, input_type, max->output().shape(), mul->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(PRelu)");

    max->input_b().connect(zero->output());
    min->input_b().connect(zero->output());
    mul->input_a().connect(min->output());
    mul->input_b().connect(alpha->output());
    add->input_a().connect(max->output());
    add->input_b().connect(mul->output());

    input_tensors_.emplace(&max->input_a(), input);
    input_tensors_.emplace(&min->input_a(), input);
    output_tensors_.emplace(output, &add->output());
}

// y = 1 / (1 + exp(-x))
void onnx_importer::convert_op_Sigmoid(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    const auto &op_name { generate_name(node) };

#if 0
    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Sigmoid)");

    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    neg->name(op_name + ".neg(Sigmoid)");

    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Sigmoid)");

    auto add = graph_.emplace<binary>(binary_add, input_type, one->output().shape(), in_shape, value_range<float>::nonnegative());
    add->name(op_name + ".add(Sigmoid)");

    auto div = graph_.emplace<binary>(binary_div, input_type, one->output().shape(), in_shape, value_range<float>::nonnegative());
    div->name(op_name + ".div(Sigmoid)");

    exp->input().connect(neg->output());
    add->input_a().connect(one->output());
    add->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&neg->input(), input);
    output_tensors_.emplace(output, &div->output());
#else
    auto s = graph_.emplace<sigmoid>(input_type, in_shape);
    s->name(op_name + ".(Sigmoid)");
    input_tensors_.emplace(&s->input(), input);
    output_tensors_.emplace(output, &s->output());
#endif
}

void onnx_importer::convert_op_HardSigmoid(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    const auto &op_name { generate_name(node) };

    // y = max(0, min(1, alpha * x + beta))
    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(0.2);
    const auto &alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(HardSigmoid)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, in_shape, alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(HardSigmoid)");

    const auto beta_value = get_attribute<float>(node, "beta").value_or(0.5);
    const auto &beta = graph_.emplace<constant>(beta_value);
    beta->name(op_name + ".beta(HardSigmoid)");

    auto sum = graph_.emplace<binary>(binary_add, input_type, mul->output().shape(), beta->output().shape(), value_range<float>::full());
    sum->name(op_name + ".sum(HardSigmoid)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSigmoid)");
    auto min = graph_.emplace<binary>(binary_min, input_type, sum->output().shape(), one->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(HardSigmoid)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSigmoid)");
    auto max = graph_.emplace<binary>(binary_max, input_type, min->output().shape(), zero->output().shape(), value_range<float>::full());
    max->name(generate_name(node) + ".max(HardSigmoid)");

    mul->input_b().connect(alpha->output());
    sum->input_a().connect(mul->output());
    sum->input_b().connect(beta->output());
    min->input_a().connect(sum->output());
    min->input_b().connect(one->output());
    max->input_a().connect(min->output());
    max->input_b().connect(zero->output());

    input_tensors_.emplace(&mul->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_HardSwish(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    const auto &op_name { generate_name(node) };

    // y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), where alpha = 1/6 and beta = 0.5
    const auto &alpha = graph_.emplace<constant>(1.0f / 6);
    alpha->name(op_name + ".alpha(HardSwish)");

    auto mul_1 = graph_.emplace<binary>(binary_mul, input_type, in_shape, alpha->output().shape(), value_range<float>::full());
    mul_1->name(op_name + ".mul_1(HardSwish)");

    const auto &beta = graph_.emplace<constant>(0.5f);
    beta->name(op_name + ".beta(HardSwish)");

    auto add = graph_.emplace<binary>(binary_add, input_type, mul_1->output().shape(), beta->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(HardSwish)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSwish)");
    auto min = graph_.emplace<binary>(binary_min, input_type, add->output().shape(), one->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(HardSwish)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSwish)");
    auto max = graph_.emplace<binary>(binary_max, input_type, min->output().shape(), zero->output().shape(), value_range<float>::full());
    max->name(generate_name(node) + ".max(HardSwish)");

    auto mul_2 = graph_.emplace<binary>(binary_mul, input_type, in_shape, max->output().shape(), value_range<float>::full());
    mul_2->name(op_name + ".mul_2(HardSwish)");

    mul_1->input_b().connect(alpha->output());
    add->input_a().connect(mul_1->output());
    add->input_b().connect(beta->output());
    min->input_a().connect(add->output());
    min->input_b().connect(one->output());
    max->input_a().connect(min->output());
    max->input_b().connect(zero->output());
    mul_2->input_b().connect(max->output());

    input_tensors_.emplace(&mul_1->input_a(), input);
    input_tensors_.emplace(&mul_2->input_a(), input);
    output_tensors_.emplace(output, &mul_2->output());
}

// Elu(x) = alpha * (exp(x) - 1.), for x < 0
// Elu(x) = x, for x >= 0
// Elu(x) can be transformed as: Elu(x) = alpha * min(exp(x) - 1.f, 0) + max(x, 0.f)
void onnx_importer::convert_op_Elu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(1.0f);
    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(Elu)");

    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Elu)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Elu)");

    auto sub = graph_.emplace<binary>(binary_sub, input_type, exp->output().shape(), one->output().shape(), value_range<float>::full());
    sub->name(op_name + ".sub(Elu)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Elu)");

    auto min = graph_.emplace<binary>(binary_min, input_type, sub->output().shape(), zero->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(Elu)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, min->output().shape(), alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(Elu)");

    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(Elu)");

    auto add = graph_.emplace<binary>(binary_add, input_type, mul->output().shape(), max->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(Elu)");

    sub->input_a().connect(exp->output());
    sub->input_b().connect(one->output());
    min->input_a().connect(sub->output());
    min->input_b().connect(zero->output());
    mul->input_a().connect(min->output());
    mul->input_b().connect(alpha->output());
    max->input_b().connect(zero->output());
    add->input_a().connect(mul->output());
    add->input_b().connect(max->output());

    input_tensors_.emplace(&exp->input(), input);
    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &add->output());
}

// Celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
void onnx_importer::convert_op_Celu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    // alpha
    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(1.0f);
    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(Celu)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Celu)");

    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, zero->output().shape(), value_range<float>::nonnegative());
    max->name(op_name + ".max(Celu)");

    auto div = graph_.emplace<binary>(binary_div, input_type, in_shape, alpha->output().shape(), value_range<float>::full());
    div->name(op_name + ".div(Celu)");

    auto exp = graph_.emplace<unary>(unary_exp, div->output().shape());
    exp->name(op_name + ".exp(Celu)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Celu)");

    auto sub = graph_.emplace<binary>(binary_sub, input_type, exp->output().shape(), one->output().shape(), value_range<float>::full());
    sub->name(op_name + ".sub(Celu)");

    auto mul = graph_.emplace<binary>(binary_mul, input_type, sub->output().shape(), alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(Celu)");

    auto min = graph_.emplace<binary>(binary_min, input_type, mul->output().shape(), zero->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(Celu)");

    auto add = graph_.emplace<binary>(binary_add, input_type, max->output().shape(), min->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(Celu)");

    max->input_b().connect(zero->output());
    div->input_b().connect(alpha->output());
    exp->input().connect(div->output());
    sub->input_a().connect(exp->output());
    sub->input_b().connect(one->output());
    mul->input_a().connect(sub->output());
    mul->input_b().connect(alpha->output());
    min->input_a().connect(mul->output());
    min->input_b().connect(zero->output());
    add->input_a().connect(max->output());
    add->input_b().connect(min->output());

    input_tensors_.emplace(&max->input_a(), input);
    input_tensors_.emplace(&div->input_a(), input);
    output_tensors_.emplace(output, &add->output());
}

// y = gamma * (alpha * e^x - alpha), for x <= 0
// y = gamma * x,                     for x > 0
// Selu(x) can be transformed as y = gamma * (alpha * min(e^x - 1, 0) + max(x, 0))
void onnx_importer::convert_op_Selu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();

    // alpha
    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(1.67326319217681884765625f);
    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(Selu)");

    // gamma
    const auto gamma_value = get_attribute<float>(node, "gamma").value_or(1.05070102214813232421875f);
    auto gamma = graph_.emplace<constant>(gamma_value);
    gamma->name(op_name + ".gamma(Selu)");

    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Selu)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Selu)");

    auto sub = graph_.emplace<binary>(binary_sub, input_type, exp->output().shape(), one->output().shape(), value_range<float>::full());
    sub->name(op_name + ".sub(Selu)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Selu)");

    auto min = graph_.emplace<binary>(binary_min, input_type, sub->output().shape(), zero->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(Selu)");

    auto mul1 = graph_.emplace<binary>(binary_mul, input_type, min->output().shape(), alpha->output().shape(), value_range<float>::full());
    mul1->name(op_name + ".mul1(Selu)");

    auto max = graph_.emplace<binary>(binary_max, input_type, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(Selu)");

    auto add = graph_.emplace<binary>(binary_add, input_type, mul1->output().shape(), max->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(Selu)");

    auto mul2 = graph_.emplace<binary>(binary_mul, input_type, add->output().shape(), gamma->output().shape(), value_range<float>::full());
    mul2->name(op_name + ".mul2(Selu)");

    sub->input_a().connect(exp->output());
    sub->input_b().connect(one->output());
    min->input_a().connect(sub->output());
    min->input_b().connect(zero->output());
    mul1->input_a().connect(min->output());
    mul1->input_b().connect(alpha->output());
    max->input_b().connect(zero->output());
    add->input_a().connect(mul1->output());
    add->input_b().connect(max->output());
    mul2->input_a().connect(add->output());
    mul2->input_b().connect(gamma->output());

    input_tensors_.emplace(&exp->input(), input);
    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &mul2->output());
}

void onnx_importer::convert_op_Trilu(const NodeProto &node)
{
    auto input_size = node.input().size();
    assert(input_size >= 1);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input = node.input()[0];
    const datatype_t input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    assert(input_shape.size() >= 2);
    const auto &output = node.output()[0];

    // upper
    bool upper = get_attribute<int>(node, "upper").value_or(1);

    // k
    int64_t k = 0;
    if (input_size > 1)
    {
        auto v = get_constant_value<int64_t>(node.input()[1]);
        assert(v.size() == 1);
        k = v[0];
    }

    auto op = graph_.emplace<trilu>(input_type, input_shape, upper, k);
    op->name(op_name + "/trilu");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}

// y = ln(exp(x) + 1)
void onnx_importer::convert_op_Softplus(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    const auto &op_name { generate_name(node) };

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Softplus)");

    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Softplus)");

    auto add = graph_.emplace<binary>(binary_add, input_type, in_shape, one->output().shape(), value_range<float>::nonnegative());
    add->name(op_name + ".add(Softplus)");

    auto log = graph_.emplace<unary>(unary_log, add->output().shape());
    log->name(op_name + ".log(Softplus)");

    add->input_a().connect(exp->output());
    add->input_b().connect(one->output());
    log->input().connect(add->output());

    input_tensors_.emplace(&exp->input(), input);
    output_tensors_.emplace(output, &log->output());
}

// y = (x / (1 + |x|))
void onnx_importer::convert_op_Softsign(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    const auto &op_name { generate_name(node) };

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(SoftSign)");

    auto abs = graph_.emplace<unary>(unary_abs, in_shape);
    abs->name(op_name + ".exp(SoftSign)");

    auto add = graph_.emplace<binary>(binary_add, input_type, in_shape, one->output().shape(), value_range<float>::nonnegative());
    add->name(op_name + ".add(SoftSign)");

    auto div = graph_.emplace<binary>(binary_div, input_type, in_shape, add->output().shape(), value_range<float>::nonnegative());
    div->name(op_name + ".div(SoftSign)");

    add->input_a().connect(abs->output());
    add->input_b().connect(one->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&abs->input(), input);
    input_tensors_.emplace(&div->input_a(), input);
    output_tensors_.emplace(output, &div->output());
}

void onnx_importer::convert_op_ThresholdedRelu(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);
    const auto input_type = get_datatype(input).value();
    const auto &op_name { generate_name(node) };

    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(1.0);
    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(ThresholdedRelu)");

    auto cmp = graph_.emplace<compare>(compare_op_t::compare_greater, input_type, in_shape, alpha->output().shape());
    cmp->name(op_name + ".greater(ThresholdedRelu)");
    cmp->input_b().connect(alpha->output());

    auto new_alpha = graph_.emplace<convert>(cmp->output().type(), cmp->output().shape(), dt_float32);
    new_alpha->name(op_name + ".new_alpha(ThresholdedRelu)");
    new_alpha->input().connect(cmp->output());

    auto b_max = graph_.emplace<binary>(binary_mul, input_type, in_shape, new_alpha->output().shape(), value_range<float>::nonnegative());
    b_max->name(op_name + ".mul(ThresholdedRelu)");

    b_max->input_b().connect(new_alpha->output());

    input_tensors_.emplace(&cmp->input_a(), input);
    input_tensors_.emplace(&b_max->input_a(), input);
    output_tensors_.emplace(output, &b_max->output());
}