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
#include <nncase/ir/ops/pad.h>
#include <sstream>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Pad(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const bool use_opset_1_or_2 = node.input().size() == 1;

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    // pad mode
    pad_mode_t mode = pad_constant;
    const auto mode_attr = get_attribute<std::string>(node, "mode");
    std::string mode_str = mode_attr ? mode_attr.value() : "constant";
    if (mode_str == "constant")
    {
        mode = pad_constant;
    }
    else if (mode_str == "reflect")
    {
        mode = pad_reflect;
    }
    else if (mode_str == "edge")
    {
        mode = pad_edge;
    }
    else
    {
        std::stringstream ss;
        ss << "Invalid pad mode: " << mode_str;
        throw std::invalid_argument(ss.str());
    }

    // pads
    axis_t paddings;
    if (use_opset_1_or_2)
    {
        // op version 1
        auto padding_attr = get_attribute<axis_t>(node, "paddings");
        if (!padding_attr)
        {
            // op version 2
            padding_attr = get_attribute<axis_t>(node, "pads");
            if (!padding_attr)
                throw std::runtime_error("\"paddings or pads\" attribute is required in Pad operator in opsets version 2 and lower");
        }

        paddings = padding_attr.value();
    }
    else
    {
        // op version 11/13
        paddings = get_constant_value<int, int64_t>(node.input()[1]);
    }

    assert(paddings.size() == 2 * input_shape.size());
    if (paddings.size() < 4)
    {
        throw std::runtime_error("Only 2D padding is supported");
    }

    const xt::svector<padding> &new_paddings = parse_padding(paddings);

    // constant_value
    scalar constant = 0.f;
    if (use_opset_1_or_2)
    {
        const auto constant_attr = get_attribute<float>(node, "value");
        constant = constant_attr ? constant_attr.value() : 0.f;
    }
    else
    {
        if (node.input().size() == 3)
        {
            const auto &constant_value = node.input()[2];
            const auto &constant_initializer = get_initializer(constant_value);
            switch (input_type)
            {
            case dt_float32:
            {
                if (constant_initializer)
                {
                    if (constant_initializer.value().float_data_size() == 0)
                    {
                        constant = 0.f;
                    }
                    else
                    {
                        constant = to<float>(constant_initializer.value());
                    }
                }
                else
                {
                    // try to extract data from previous constant nodes
                    const auto data { get_constant_input_data<float>(constant_value) };

                    if (data && data.value().size())
                    {
                        constant = data.value()[0];
                    }
                }
                break;
            }

            case dt_uint8:
            {
                if (constant_initializer)
                {
                    constant = to<uint8_t>(*constant_initializer);
                }
                else
                {
                    // try to extract data from previous constant nodes
                    const auto data = get_constant_input_data<uint8_t>(constant_value);

                    if (data && data.value().size())
                    {
                        constant = data.value()[0];
                    }
                }
                break;
            }
            default:
                break;
            }
        }
    }

    auto op = graph_.emplace<pad>(input_type, input_shape, new_paddings, mode, std::move(constant));
    op->name(op_name + "(Pad)");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
