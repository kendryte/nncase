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
#include <limits>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/reduce_window2d.h>

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

void onnx_importer::convert_op_AveragePool(const NodeProto &node)
{
    convert_pool<>(node, reduce_mean, 0.f);
}

void onnx_importer::convert_op_GlobalAveragePool(const NodeProto &node)
{
    convert_pool<true>(node, reduce_mean, 0.f);
}

void onnx_importer::convert_op_MaxPool(const NodeProto &node)
{
    convert_pool<>(node, reduce_max, std::numeric_limits<float>::lowest());
}

void onnx_importer::convert_op_GlobalMaxPool(const NodeProto &node)
{
    convert_pool<true>(node, reduce_max, std::numeric_limits<float>::lowest());
}

template <bool global>
void onnx_importer::convert_pool(const NodeProto &node, const reduce_op_t reduce_op, const float init_value)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);
    padding_mode pad_mode = padding_mode::notset;

    const auto &auto_pad_attr = get_attribute<std::string>(node, "auto_pad");
    if (auto_pad_attr)
    {
        pad_mode = parse_padding_mode(auto_pad_attr.value());
    }

    std::array<size_t, 2> dilations = { 1, 1 };

    if (input_shape.size() < 4)
        throw std::invalid_argument("Image with 4-dimensional shape is expected on the input of pooling operators.");

    const auto &kernel_shape = global ? std::vector<int> { static_cast<int>(input_shape[2]), static_cast<int>(input_shape[3]) } :
                                        get_attribute<std::vector<int>>(node, "kernel_shape").value();

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

    std::vector<padding> pads {
        { 0, 0 },
        { 0, 0 }
    };

    switch (pad_mode)
    {
    case padding_mode::notset:
    {
        const auto &pads_attr = get_attribute<axis_t>(node, "pads");
        if (pads_attr)
            pads = parse_padding(pads_attr.value());

        break;
    }
    case padding_mode::same:
    {
        pads[0] = get_windowed_padding(input_shape[2], kernel_shape[0], strides[0], dilations[0], true);
        pads[1] = get_windowed_padding(input_shape[3], kernel_shape[1], strides[1], dilations[1], true);
        break;
    }
    case padding_mode::valid:
    {
        break;
    }
    }

    auto op = graph_.emplace<reduce_window2d>(reduce_op, move(input_shape), init_value, kernel_shape[0], kernel_shape[1],
                                              pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], value_range<float>::full());

    op->name(op_name + '.' + reduce_op_to_string(reduce_op)+ "(Pool)");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
