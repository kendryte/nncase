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

#include <hlir/graph.h>
#include <hlir/op_utils.h>
#include <hlir/ops/conv2d.h>
#include <hlir/ops/conv2d_transpose.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    enum class padding_mode
    {
        notset,
        same,
        valid
    };

    padding_mode parse_padding_mode(const string &value) noexcept
    {
        if (value == "VALID")
            return padding_mode::valid;
        else if (value == "SAME_UPPER" || value == "SAME_LOWER")
            return padding_mode::same;
        else
            return padding_mode::notset;
    }

    shape_t generate_output_shape(const shape_t& input, const shape_t& kernel, const array<padding, 2>& pads, const array<size_t, 2>& dilations, const array<size_t, 2>& strides)
    {
        return
        {
            input[0],
            kernel[1],
            input[2] + dilations[0] * (kernel[2] - 1) - pads[0].sum(),
            input[3] + dilations[1] * (kernel[3] - 1) - pads[1].sum()
        };
    }
}

void onnx_importer::convert_op_Conv(const NodeProto& node)
{
    convert_conv<conv2d>(node);
}

void onnx_importer::convert_op_ConvTranspose(const NodeProto& node)
{
    convert_conv<conv2d_transpose>(node);
}

template<class Node> void onnx_importer::convert_conv(const NodeProto &node)
{
    const auto &input { node.input()[0] };
    const auto &weight { node.input()[1] };
    const auto &output { node.output()[0] };

    auto input_shape { get_shape(input) };

    padding_mode pad_mode { padding_mode::notset };

    const auto &auto_pad_attr { get_attribute<string>(node, "auto_pad") };
    if (auto_pad_attr)
    {
        pad_mode = parse_padding_mode(auto_pad_attr.value());
    }

    array<size_t, 2> dilations { 1, 1 };

    const auto &dilations_attr { get_attribute<vector<int>>(node, "dilations") };
    if (dilations_attr)
    {
        const auto &dilations_values { dilations_attr.value() };

        if (dilations_values.size() > 0)
            dilations[0] = dilations_values[0];
        if (dilations_values.size() > 1)
            dilations[1] = dilations_values[1];
    }

    size_t group { 1 };

    const auto &group_attr { get_attribute<int>(node, "group") };
    if (group_attr)
        group = group_attr.value();

    array<size_t, 2> strides { 1, 1 };

    const auto &strides_attr { get_attribute<vector<int>>(node, "strides") };
    if (strides_attr)
    {
        const auto &strides_values { strides_attr.value() };

        if (strides_values.size() > 0)
            strides[0] = strides_values[0];
        if (strides_values.size() > 1)
            strides[1] = strides_values[1];
    }

    const auto &weight_initializer { get_initializer(weight) };

    if (!weight_initializer)
        throw runtime_error("Can't find initializer for weight input");

    const auto &weight_shape { get_shape(weight_initializer.value()) };

    array<padding, 2> pads
    {{
        { 0, 0 },
        { 0, 0 }
    }};

    switch (pad_mode)
    {
    case padding_mode::notset:
    {
        const auto &pads_attr { get_attribute<vector<int>>(node, "pads") };

        if (pads_attr)
        {
            const auto &pads_values { pads_attr.value() };
            if (pads_values.size() > 1)
            {
                pads[0].before = pads_values[0];
                pads[0].after = pads_values[1];
            }

            if (pads_values.size() > 3)
            {
                pads[1].before = pads_values[2];
                pads[1].after = pads_values[3];
            }
        }

        break;
    }
    case padding_mode::same:
        pads[0] = get_windowed_padding(input_shape[2], weight_shape[2], strides[0], dilations[0], true);
        pads[1] = get_windowed_padding(input_shape[3], weight_shape[3], strides[1], dilations[1], true);
        break;
    }

    auto &&weight_value { to<xt::xarray<float>>(weight_initializer.value()) };

    xt::xarray<float> &&bias_value { xt::zeros<float>({ weight_shape[0] }) };
    if (node.input().size() > 2)
    {
        const auto &bias { node.input()[2] };
        const auto &bias_initializer { get_initializer(bias) };
        if (!bias_initializer)
            throw runtime_error("Can't find initializer for bias input");

        bias_value = to<xt::xarray<float>>(bias_initializer.value());
    }

    auto conv { add_conv_node<Node>(node, graph_, move(input_shape), move(weight_value), move(bias_value), group, pads, strides, dilations) };

    input_tensors_.emplace(&conv->input(), input);
    output_tensors_.emplace(output, &conv->output());
}

template<class Node> Node* onnx_importer::add_conv_node(const NodeProto &node, hlir::graph& graph, shape_t&& input_shape, xt::xarray<float>&& weight_value, xt::xarray<float>&& bias_value, const size_t group, const array<padding, 2>& pads, const array<size_t, 2>& strides, const array<size_t, 2>& dilations)
{
    return graph.emplace<Node>(move(input_shape), move(weight_value), move(bias_value), group, pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], value_range<float>::full());
}

template<> conv2d_transpose* onnx_importer::add_conv_node<conv2d_transpose>(const NodeProto &node, hlir::graph& graph, shape_t&& input_shape, xt::xarray<float>&& weight_value, xt::xarray<float>&& bias_value, const size_t group, const array<padding, 2>& pads, const array<size_t, 2>& strides, const array<size_t, 2>& dilations)
{
    auto output_shape { generate_output_shape(input_shape, weight_value.shape(), pads, dilations, strides) };

    const auto &output_shape_attr { get_attribute<vector<int>>(node, "output_shape") };
    if (output_shape_attr)
    {
        const auto &output_shape_value { output_shape_attr.value() };
        output_shape = shape_t { begin(output_shape_value), end(output_shape_value) };
    }

    return graph.emplace<conv2d_transpose>(move(input_shape), move(output_shape), move(weight_value), move(bias_value), group, pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], value_range<float>::full());
}
