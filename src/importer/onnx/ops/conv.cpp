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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/conv2d_transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

namespace
{
enum class padding_mode
{
    notset,
    same,
    valid
};

padding_mode parse_padding_mode(const std::string &value) noexcept
{
    if (value == "VALID")
        return padding_mode::valid;
    else if (value == "SAME_UPPER" || value == "SAME_LOWER")
        return padding_mode::same;
    else
        return padding_mode::notset;
}

shape_t generate_output_shape(const shape_t &input, const shape_t &kernel, const std::array<padding, 2> &pads, const std::array<size_t, 2> &dilations,
    [[maybe_unused]] const std::array<size_t, 2> &strides)
{
    return {
        input[0],
        kernel[1],
        input[2] + dilations[0] * (kernel[2] - 1) - pads[0].sum(),
        input[3] + dilations[1] * (kernel[3] - 1) - pads[1].sum()
    };
}
}

void onnx_importer::convert_op_Conv(const NodeProto &node)
{
    convert_conv<conv2d>(node);
}

void onnx_importer::convert_op_ConvTranspose(const NodeProto &node)
{
    convert_conv<conv2d_transpose>(node);
}

template <class Node>
void onnx_importer::convert_conv(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &weight = node.input()[1];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);
    auto weight_shape = get_shape(weight);

    // group
    size_t group = 1;
    const auto &group_attr = get_attribute<int>(node, "group");
    if (group_attr)
    {
        group = group_attr.value();
    }

    // stride
    std::array<size_t, 2> strides = { 1, 1 };
    const auto &strides_attr = get_attribute<std::vector<int>>(node, "strides");
    if (strides_attr)
    {
        const auto &strides_values = strides_attr.value();
        if (strides_values.size() > 0)
            strides[0] = strides_values[0];
        if (strides_values.size() > 1)
            strides[1] = strides_values[1];
    }

    // dilations
    std::array<size_t, 2> dilations = { 1, 1 };
    const auto &dilations_attr = get_attribute<std::vector<int>>(node, "dilations");
    if (dilations_attr)
    {
        const auto &dilations_values = dilations_attr.value();
        if (dilations_values.size() > 0)
            dilations[0] = dilations_values[0];
        if (dilations_values.size() > 1)
            dilations[1] = dilations_values[1];
    }

    // pad
    std::array<padding, 2> pads { { { 0, 0 }, { 0, 0 } } };
    padding_mode pad_mode = padding_mode::notset;
    const auto &auto_pad_attr = get_attribute<std::string>(node, "auto_pad");
    if (auto_pad_attr)
    {
        pad_mode = parse_padding_mode(auto_pad_attr.value());
    }
    switch (pad_mode)
    {
    case padding_mode::notset:
    {
        const auto &pads_attr = get_attribute<std::vector<int>>(node, "pads");

        if (pads_attr)
        {
            const auto &pads_values = pads_attr.value();
            if (pads_values.size() > 1)
            {
                pads[0].before = pads_values[0];
                pads[1].before = pads_values[1];
            }

            if (pads_values.size() > 3)
            {
                pads[0].after = pads_values[2];
                pads[1].after = pads_values[3];
            }
        }

        break;
    }
    case padding_mode::same:
    {
        pads[0] = get_windowed_padding(input_shape[2], weight_shape[2], strides[0], dilations[0], true);
        pads[1] = get_windowed_padding(input_shape[3], weight_shape[3], strides[1], dilations[1], true);
        break;
    }
    default:
        break;
    }

    auto conv = add_conv_node<Node>(node, graph_, input_shape, weight_shape, group, pads, strides, dilations);
    conv->name(op_name + "(Conv2d)");

    input_tensors_.emplace(&conv->input(), input);
    input_tensors_.emplace(&conv->weights(), weight);
    if (node.input().size() > 2)
    {
        const auto &bias = node.input()[2];
        input_tensors_.emplace(&conv->bias(), bias);
    }
    else
    {
        std::vector<float> bias_value(weight_shape[0], 0.f);
        shape_t bias_shape = { weight_shape[0] };
        auto bias_node = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
        conv->bias().connect(bias_node->output());
    }
    output_tensors_.emplace(output, &conv->output());
}

template <class Node>
Node *onnx_importer::add_conv_node([[maybe_unused]] const NodeProto &node, ir::graph &graph, shape_t input_shape, ir::shape_t weight_shape,
    const size_t group, const std::array<padding, 2> &pads, const std::array<size_t, 2> &strides, const std::array<size_t, 2> &dilations)
{
    return graph.emplace<Node>(input_shape, weight_shape, group, pads[0], pads[1], strides[0], strides[1],
        dilations[0], dilations[1], value_range<float>::full());
}

template <>
conv2d_transpose *onnx_importer::add_conv_node<conv2d_transpose>([[maybe_unused]] const NodeProto &node, ir::graph &graph, shape_t input_shape, ir::shape_t weight_shape, const size_t group,
    const std::array<padding, 2> &pads, const std::array<size_t, 2> &strides, const std::array<size_t, 2> &dilations)
{
    auto output_shape = generate_output_shape(input_shape, weight_shape, pads, dilations, strides);
    const auto &output_shape_attr = get_attribute<std::vector<int>>(node, "output_shape");
    if (output_shape_attr)
    {
        const auto &output_shape_value = output_shape_attr.value();
        output_shape = shape_t { std::begin(output_shape_value), std::end(output_shape_value) };
    }

    return graph.emplace<conv2d_transpose>(input_shape, weight_shape, output_shape, group, pads[0], pads[1], strides[0], strides[1],
        dilations[0], dilations[1], value_range<float>::full());
}
