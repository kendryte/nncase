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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/reduce.h>
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

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Relu)");
    auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());
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

    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(0.01f);
    const auto &alpha = graph_.emplace<constant>(alpha_value);

    alpha->name(op_name + ".alpha(LeakyRelu)");

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(LeakyRelu)");
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());
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

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(PRelu)");
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(PRelu)");

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&mul->input_a(), input);
    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_Sigmoid(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);

    const auto &op_name { generate_name(node) };

#if 0
    // y = 1 / (1 + exp(-x))
    auto one = graph_.emplace<constant>(1.f);
    auto neg = graph_.emplace<unary>(unary_neg, in_shape);
    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    auto add = graph_.emplace<binary>(binary_add, one->output().shape(), in_shape, value_range<float>::nonnegative());
    auto div = graph_.emplace<binary>(binary_div, one->output().shape(), in_shape, value_range<float>::nonnegative());
    exp->input().connect(neg->output());
    add->input_a().connect(one->output());
    add->input_b().connect(exp->output());
    div->input_a().connect(one->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&neg->input(), input);
    output_tensors_.emplace(output, &div->output());
#else
    // y = exp(x) / (exp(x) + 1)
    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Sigmoid)");
    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Sigmoid)");
    auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
    add->name(op_name + ".add(Sigmoid)");
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), add->output().shape(), value_range<float>::nonnegative());
    div->name(op_name + ".div(Sigmoid)");
    add->input_a().connect(exp->output());
    add->input_b().connect(one->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&exp->input(), input);
    output_tensors_.emplace(output, &div->output());
#endif
}

void onnx_importer::convert_op_HardSigmoid(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto in_shape = get_shape(input);

    const auto &op_name { generate_name(node) };

    // y = max(0, min(1, alpha * x + beta))
    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(0.2);
    const auto &alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(HardSigmoid)");

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    mul->name(op_name + ".mul(HardSigmoid)");

    const auto beta_value = get_attribute<float>(node, "beta").value_or(0.5);
    const auto &beta = graph_.emplace<constant>(beta_value);
    beta->name(op_name + ".beta(HardSigmoid)");

    auto sum = graph_.emplace<binary>(binary_add, mul->output().shape(), beta->output().shape(), value_range<float>::full());
    sum->name(op_name + ".sum(HardSigmoid)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSigmoid)");
    auto min = graph_.emplace<binary>(binary_min, sum->output().shape(), one->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(HardSigmoid)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSigmoid)");
    auto max = graph_.emplace<binary>(binary_max, min->output().shape(), zero->output().shape(), value_range<float>::full());
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

    const auto &op_name { generate_name(node) };

    // y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), where alpha = 1/6 and beta = 0.5
    const auto &alpha = graph_.emplace<constant>(1.0f / 6);
    alpha->name(op_name + ".alpha(HardSwish)");

    auto mul_1 = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    mul_1->name(op_name + ".mul_1(HardSwish)");

    const auto &beta = graph_.emplace<constant>(0.5f);
    beta->name(op_name + ".beta(HardSwish)");

    auto add = graph_.emplace<binary>(binary_add, mul_1->output().shape(), beta->output().shape(), value_range<float>::full());
    add->name(op_name + ".add(HardSwish)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(HardSwish)");
    auto min = graph_.emplace<binary>(binary_min, add->output().shape(), one->output().shape(), value_range<float>::full());
    min->name(op_name + ".min(HardSwish)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(HardSwish)");
    auto max = graph_.emplace<binary>(binary_max, min->output().shape(), zero->output().shape(), value_range<float>::full());
    max->name(generate_name(node) + ".max(HardSwish)");

    auto mul_2 = graph_.emplace<binary>(binary_mul, in_shape, max->output().shape(), value_range<float>::full());
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

    const auto alpha_value = get_attribute<float>(node, "alpha").value_or(1.0f);
    auto alpha = graph_.emplace<constant>(alpha_value);
    alpha->name(op_name + ".alpha(Elu)");

    auto exp = graph_.emplace<unary>(unary_exp, in_shape);
    exp->name(op_name + ".exp(Elu)");

    auto one = graph_.emplace<constant>(1.f);
    one->name(op_name + ".one(Elu)");

    auto sub = graph_.emplace<binary>(binary_sub, exp->output().shape(), one->output().shape(), value_range<float>::full());
    sub->name(generate_name(node) + ".sub(Elu)");

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(Elu)");

    auto min = graph_.emplace<binary>(binary_min, sub->output().shape(), zero->output().shape(), value_range<float>::full());
    min->name(generate_name(node) + ".min(Elu)");

    auto mul = graph_.emplace<binary>(binary_mul, min->output().shape(), alpha->output().shape(), value_range<float>::full());
    mul->name(generate_name(node) + ".mul(Elu)");

    auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(generate_name(node) + ".max(Elu)");

    auto add = graph_.emplace<binary>(binary_add, mul->output().shape(), max->output().shape(), value_range<float>::full());
    add->name(generate_name(node) + ".add(Elu)");

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
