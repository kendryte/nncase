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

void onnx_importer::convert_op_Relu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    auto in_shape = get_shape(input);

    auto zero = graph_.emplace<constant>(0.f);
    auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_LeakyRelu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    auto &&in_shape = get_shape(input);

    const auto alpha_value = get_attribute<float>(node, "alpha").value();
    const auto &alpha = graph_.emplace<constant>(alpha_value);

    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());

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
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());

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
    auto one = graph_.emplace<constant>(1.f);
    auto add = graph_.emplace<binary>(binary_add, exp->output().shape(), one->output().shape(), value_range<float>::nonnegative());
    auto div = graph_.emplace<binary>(binary_div, exp->output().shape(), add->output().shape(), value_range<float>::nonnegative());
    add->input_a().connect(exp->output());
    add->input_b().connect(one->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(add->output());

    input_tensors_.emplace(&exp->input(), input);
    output_tensors_.emplace(output, &div->output());
#endif
}