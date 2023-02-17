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
#include <nncase/ir/ir_types.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/conv2d_transpose.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Conv(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &weight = node.input()[1];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);
    auto weight_shape = get_shape(weight);
    auto output_shape = get_shape(output);
    auto input_type = get_datatype(input).value();
    auto weights_type = get_datatype(weight).value();
    auto output_type = get_datatype(output).value();
    // group
    const auto &group_attr = get_attribute<int>(node, "group");
    size_t group = group_attr ? group_attr.value() : 1;

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
    std::array<padding, 2> paddings { { { 0, 0 }, { 0, 0 } } };
    const auto &auto_pad_attr = get_attribute<std::string>(node, "auto_pad");
    std::string pad_mode = auto_pad_attr ? auto_pad_attr.value() : "NOTSET";
    if (pad_mode == "NOTSET")
    {
        const auto &paddings_attr = get_attribute<std::vector<int>>(node, "pads");
        if (paddings_attr)
        {
            const auto &paddings_values = paddings_attr.value();
            if (paddings_values.size() == 2)
            {
                paddings[0].before = paddings_values[0];
                paddings[0].after = paddings_values[1];
            }
            else if (paddings_values.size() == 4)
            {
                paddings[0].before = paddings_values[0];
                paddings[1].before = paddings_values[1];
                paddings[0].after = paddings_values[2];
                paddings[1].after = paddings_values[3];
            }
        }
    }
    else if (pad_mode == "SAME_UPPER")
    {
        paddings[0] = get_windowed_padding(input_shape[2], weight_shape[2], strides[0], dilations[0], true);
        paddings[1] = get_windowed_padding(input_shape[3], weight_shape[3], strides[1], dilations[1], true);
    }
    else if (pad_mode == "SAME_LOWER")
    {
        paddings[0] = get_windowed_padding(input_shape[2], weight_shape[2], strides[0], dilations[0], true);
        if (paddings[0].before < paddings[0].after)
            std::swap(paddings[0].before, paddings[0].after);

        paddings[1] = get_windowed_padding(input_shape[3], weight_shape[3], strides[1], dilations[1], true);
        if (paddings[1].before < paddings[1].after)
            std::swap(paddings[1].before, paddings[1].after);
    }

    // fit 3D input
    bitcast *bitc_data, *bitc_weights;
    bool model_3d = false;
    if (input_shape.size() == 3)
    {
        model_3d = true;
        paddings[1] = padding::zero();
        strides[1] = 1;
        dilations[1] = 1;
        auto data_shape = input_shape;
        input_shape.push_back(1);
        bitc_data = graph_.emplace<bitcast>(input_type, data_shape, input_shape);

        auto weights_shape = weight_shape;
        weight_shape.push_back(1);
        bitc_weights = graph_.emplace<bitcast>(weights_type, weights_shape, weight_shape);
    }

    auto conv = graph_.emplace<conv2d>(input_shape, weight_shape, group, paddings[0], paddings[1], strides[0], strides[1],
        dilations[0], dilations[1], value_range<float>::full());
    conv->name(generate_name(node) + "(Conv)");

    if (model_3d)
    {
        conv->input().connect(bitc_data->output());
        input_tensors_.emplace(&bitc_data->input(), input);

        conv->weights().connect(bitc_weights->output());
        input_tensors_.emplace(&bitc_weights->input(), weight);
    }
    else
    {
        input_tensors_.emplace(&conv->input(), input);
        input_tensors_.emplace(&conv->weights(), weight);
    }

    if (node.input().size() > 2)
    {
        input_tensors_.emplace(&conv->bias(), node.input()[2]);
    }
    else
    {
        shape_t shape = { weight_shape[0] };
        std::vector<float> zeros(weight_shape[0], 0.f);
        auto bias = graph_.emplace<constant>(dt_float32, shape, zeros);
        conv->bias().connect(bias->output());
    }

    if (model_3d)
    {
        auto bitc_out = graph_.emplace<bitcast>(output_type, conv->output().shape(), shape_t { conv->output().shape()[0], conv->output().shape()[1], conv->output().shape()[2] });
        bitc_out->input().connect(conv->output());
        output_tensors_.emplace(output, &bitc_out->output());
    }
    else
        output_tensors_.emplace(output, &conv->output());
}

void onnx_importer::convert_op_ConvTranspose(const NodeProto &node)
{
    const auto &op_name = generate_name(node);
    const auto &input = node.input()[0];
    const auto &weight = node.input()[1];
    const auto &output = node.output()[0];

    auto input_shape = get_shape(input);
    auto weight_shape = get_shape(weight);
    auto output_shape = get_shape(output);
    auto input_type = get_datatype(input).value();
    auto weight_type = get_datatype(weight).value();
    auto output_type = get_datatype(output).value();

    bool model_3d = input_shape.size() == 3;

    // group
    const auto &group_attr = get_attribute<int>(node, "group");
    size_t group = group_attr ? group_attr.value() : 1;

    transpose *tp;
    bitcast *bc;
    shape_t bc_shape, tp_shape;
    if (model_3d)
    {
        tp = graph_.emplace<transpose>(weight_type, weight_shape, axis_t { 1, 0, 2 });
        tp->name(op_name + "(Transpose)");
        tp_shape = tp->output().shape();
        bc = graph_.emplace<bitcast>(weight_type, tp_shape, shape_t { tp_shape[0] * group, tp_shape[1] / group, tp_shape[2], 1 });
        bc->name(op_name + "(Bitcast)");
        bc_shape = bc->output().shape();
    }
    else
    {
        tp = graph_.emplace<transpose>(weight_type, weight_shape, axis_t { 1, 0, 2, 3 });
        tp->name(op_name + "(Transpose)");
        tp_shape = tp->output().shape();
        bc = graph_.emplace<bitcast>(weight_type, tp_shape, shape_t { tp_shape[0] * group, tp_shape[1] / group, tp_shape[2], tp_shape[3] });
        bc->name(op_name + "(Bitcast)");
        bc_shape = bc->output().shape();
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

    // output_padding
    std::array<int32_t, 2> output_paddings = { 0, 0 };
    const auto &output_padding_attr = get_attribute<std::vector<int>>(node, "output_padding");
    if (output_padding_attr)
    {
        const auto &output_padding_values = output_padding_attr.value();
        if (output_padding_values.size() > 0)
        {
            output_paddings[0] = output_padding_values[0];
            assert(output_paddings[0] < strides[0] || output_paddings[0] < dilations[0]);
        }

        if (output_padding_values.size() > 1)
        {
            output_paddings[1] = output_padding_values[1];
            assert(output_paddings[1] < strides[1] || output_paddings[1] < dilations[1]);
        }
    }

    // pad
    std::array<padding, 2> paddings { { { 0, 0 }, { 0, 0 } } };
    const auto &auto_pad_attr = get_attribute<std::string>(node, "auto_pad");
    std::string pad_mode = auto_pad_attr ? auto_pad_attr.value() : "NOTSET";

    // output_shape
    const auto &output_shape_attr = get_attribute<std::vector<int>>(node, "output_shape");
    if (output_shape_attr)
    {
        std::array<int, 2> total_paddings { { 0, 0 } };
        total_paddings[0] = strides[0] * (input_shape[2] - 1) + output_paddings[0] + ((tp_shape[2] - 1) * dilations[0] + 1) - output_shape[2];
        if (!model_3d)
            total_paddings[1] = strides[1] * (input_shape[3] - 1) + output_paddings[1] + ((tp_shape[3] - 1) * dilations[1] + 1) - output_shape[3];

        if (pad_mode == "SAME_UPPER")
        {
            paddings[0].before = total_paddings[0] / 2;
            paddings[0].after = total_paddings[0] - paddings[0].before;
            paddings[1].before = total_paddings[1] / 2;
            paddings[1].after = total_paddings[1] - paddings[1].before;
        }
        else
        {
            paddings[0].after = total_paddings[0] / 2;
            paddings[0].before = total_paddings[0] - paddings[0].after;
            paddings[1].after = total_paddings[1] / 2;
            paddings[1].before = total_paddings[1] - paddings[1].after;
        }
    }
    else
    {
        if (pad_mode == "NOTSET")
        {
            const auto &paddings_attr = get_attribute<std::vector<int>>(node, "pads");
            if (paddings_attr)
            {
                const auto &paddings_values = paddings_attr.value();
                if (model_3d)
                {
                    if (paddings_values.size() > 1)
                    {
                        paddings[0].before = paddings_values[0];
                        paddings[0].after = paddings_values[1];
                    }
                }
                else
                {
                    if (paddings_values.size() > 1)
                    {
                        paddings[0].before = paddings_values[0];
                        paddings[1].before = paddings_values[1];
                    }

                    if (paddings_values.size() > 3)
                    {
                        paddings[0].after = paddings_values[2];
                        paddings[1].after = paddings_values[3];
                    }
                }
            }
        }
        else if (pad_mode == "SAME_UPPER")
        {
            paddings[0] = get_windowed_padding(input_shape[2], tp_shape[2], strides[0], dilations[0], true);
            if (!model_3d)
                paddings[1] = get_windowed_padding(input_shape[3], tp_shape[3], strides[1], dilations[1], true);
        }
        else if (pad_mode == "SAME_LOWER")
        {
            paddings[0] = get_windowed_padding(input_shape[2], tp_shape[2], strides[0], dilations[0], true);
            if (paddings[0].before < paddings[0].after)
                std::swap(paddings[0].before, paddings[0].after);

            if (!model_3d)
            {
                paddings[1] = get_windowed_padding(input_shape[3], tp_shape[3], strides[1], dilations[1], true);
                if (paddings[1].before < paddings[1].after)
                    std::swap(paddings[1].before, paddings[1].after);
            }
        }
    }

    // fit 3D input
    auto data_shape = input_shape;
    if (model_3d)
    {
        paddings[1] = padding::zero();
        strides[1] = 1;
        dilations[1] = 1;
        input_shape.push_back(1);

        output_shape.push_back(1);
    }

    // ConvTranspose
    auto conv_transpose = graph_.emplace<conv2d_transpose>(input_shape, bc_shape, output_shape, group, paddings[0], paddings[1],
        output_paddings[0], output_paddings[1], strides[0], strides[1], dilations[0], dilations[1], value_range<float>::full());
    conv_transpose->name(op_name + "(ConvTranspose)");

    if (model_3d)
    {
        auto bitc_data = graph_.emplace<bitcast>(input_type, data_shape, input_shape);
        conv_transpose->input().connect(bitc_data->output());
        input_tensors_.emplace(&bitc_data->input(), input);

        input_tensors_.emplace(&tp->input(), weight);
    }
    else
    {
        input_tensors_.emplace(&conv_transpose->input(), input);
        input_tensors_.emplace(&tp->input(), weight);
    }

    bc->input().connect(tp->output());
    conv_transpose->weights().connect(bc->output());
    if (node.input().size() > 2)
    {
        input_tensors_.emplace(&conv_transpose->bias(), node.input()[2]);
    }
    else
    {
        shape_t shape = { bc_shape[0] };
        std::vector<float> zeros(bc_shape[0], 0.f);
        auto bias = graph_.emplace<constant>(dt_float32, shape, zeros);
        conv_transpose->bias().connect(bias->output());
    }

    if (model_3d)
    {
        auto bitc_out = graph_.emplace<bitcast>(output_type, conv_transpose->output().shape(), shape_t { conv_transpose->output().shape()[0], conv_transpose->output().shape()[1], conv_transpose->output().shape()[2] });
        bitc_out->input().connect(conv_transpose->output());
        output_tensors_.emplace(output, &bitc_out->output());
    }
    else
    {
        output_tensors_.emplace(output, &conv_transpose->output());
    }
}