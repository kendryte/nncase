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
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/ops/reduce.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

// y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.
void onnx_importer::convert_op_InstanceNormalization(const NodeProto &node)
{
    assert(node.input().size() == 3);

    const auto &input = node.input()[0];
    const auto &scale = node.input()[1];
    const auto &bias = node.input()[2];
    const auto output = node.output()[0];

    auto input_shape = get_shape(input);

    std::vector<float> scale_value;
    auto scale_initializer = get_initializer(scale);
    scale_value = scale_initializer ? to<std::vector<float>>(scale_initializer.value()) : get_constant_input_data<float>(scale).value();
    auto scale_shape = get_shape(scale);
    auto scale_new_shape = broadcast_shape(scale_shape, input_shape);
    auto scale_constant = graph_.emplace<constant>(get_datatype<float>(), scale_new_shape, scale_value);

    std::vector<float> bias_value;
    auto bias_initializer = get_initializer(bias);
    bias_value = bias_initializer ? to<std::vector<float>>(bias_initializer.value()) : get_constant_input_data<float>(bias).value();
    auto bias_shape = get_shape(bias);
    auto bias_new_shape = broadcast_shape(bias_shape, input_shape);
    auto bias_constant = graph_.emplace<constant>(get_datatype<float>(), bias_new_shape, bias_value);

    // mean
    axis_t axes;
    for (size_t i = 2; i < input_shape.size(); i++)
    {
        axes.push_back(i);
    }
    float init_value = 0.f;
    bool keepdims = true;
    auto mean = graph_.emplace<reduce>(reduce_mean, input_shape, axes, init_value, keepdims);

    // x - mean
    auto sub = graph_.emplace<binary>(binary_sub, input_shape, mean->output().shape(), value_range<float>::full());

    // scale * (x - mean)
    auto mul = graph_.emplace<binary>(binary_mul, scale_new_shape, sub->output().shape(), value_range<float>::full());

    // variance
    auto square = graph_.emplace<unary>(unary_square, sub->output().shape());
    auto variance = graph_.emplace<reduce>(reduce_mean, square->output().shape(), axes, init_value, keepdims);

    // sqrt(variance + epsilon)
    auto epsilon_attr = get_attribute<float>(node, "epsilon");
    auto epsilon = epsilon_attr ? epsilon_attr.value() : 1e-05f;
    auto eps_constant = graph_.emplace<constant>(epsilon);
    auto add_eps = graph_.emplace<binary>(binary_add, variance->output().shape(), eps_constant->output().shape(), value_range<float>::full());
    auto sqrt = graph_.emplace<unary>(unary_sqrt, add_eps->output().shape());

    // scale * (x - mean) / sqrt(variance + epsilon) + B
    auto div = graph_.emplace<binary>(binary_div, mul->output().shape(), sqrt->output().shape(), value_range<float>::full());
    auto add_bias = graph_.emplace<binary>(binary_add, div->output().shape(), bias_new_shape, value_range<float>::full());

    sub->input_b().connect(mean->output());

    mul->input_a().connect(scale_constant->output());
    mul->input_b().connect(sub->output());

    square->input().connect(sub->output());
    variance->input().connect(square->output());

    add_eps->input_a().connect(variance->output());
    add_eps->input_b().connect(eps_constant->output());

    sqrt->input().connect(add_eps->output());

    div->input_a().connect(mul->output());
    div->input_b().connect(sqrt->output());

    add_bias->input_a().connect(div->output());
    add_bias->input_b().connect(bias_constant->output());

    input_tensors_.emplace(&mean->input(), input);
    input_tensors_.emplace(&sub->input_a(), input);
    output_tensors_.emplace(output, &add_bias->output());
}