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
#include <nncase/ir/ops/transpose.h>

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
}

void onnx_importer::convert_op_Conv(const NodeProto &node)
{
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

    auto conv = graph_.emplace<conv2d>(input_shape, weight_shape, group, pads[0], pads[1], strides[0], strides[1],
        dilations[0], dilations[1], value_range<float>::full());
    conv->name(generate_name(node) + "(Conv)");

    input_tensors_.emplace(&conv->input(), input);
    input_tensors_.emplace(&conv->weights(), weight);
    std::cout << " node.input().size() = " << node.input().size() << std::endl;
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

void onnx_importer::convert_op_ConvTranspose(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &weight = node.input()[1];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);
    auto output_shape = get_shape(output);

    auto tp = graph_.emplace<transpose>(dt_float32, get_shape(weight), axis_t { 1, 0, 2, 3 });
    auto tp_shape = tp->output().shape();

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
        pads[0] = get_windowed_padding(input_shape[2], tp_shape[2], strides[0], dilations[0], true);
        pads[1] = get_windowed_padding(input_shape[3], tp_shape[3], strides[1], dilations[1], true);
        break;
    }
    default:
        break;
    }

    auto conv_transpose = graph_.emplace<conv2d_transpose>(input_shape, tp_shape, output_shape, group, pads[0], pads[1], strides[0], strides[1],
        dilations[0], dilations[1], value_range<float>::full());
    conv_transpose->name(generate_name(node) + "(ConvTranspose)");

    input_tensors_.emplace(&conv_transpose->input(), input);
    input_tensors_.emplace(&tp->input(), weight);
    conv_transpose->weights().connect(tp->output());
    if (node.input().size() > 2)
    {
        const auto &bias = node.input()[2];
        input_tensors_.emplace(&conv_transpose->bias(), bias);
    }
    else
    {
        std::vector<float> bias_value(tp_shape[0], 0.f);
        shape_t bias_shape = { tp_shape[0] };
        auto bias_node = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
        conv_transpose->bias().connect(bias_node->output());
    }
    output_tensors_.emplace(output, &conv_transpose->output());
}
