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
#include <hlir/ops/pad.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Pad(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const bool use_opset_version_9  { node.input().size() == 1 };

    const auto input_type { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    constexpr char constant_mode_caption[] { "constant" };
    string mode { constant_mode_caption };

    const auto mode_attr { get_attribute<string>(node, "mode") };
    if (mode_attr)
    {
        mode = mode_attr.value();
        if (mode != constant_mode_caption)
        {
            cout << "Warning: only 'constant' padding mode is supported by hardware, falling back to it" << endl;
        }
    }

    axis_t padding_value;

    if (!use_opset_version_9)
    {
        const auto &pads { node.input()[1] };
        const auto &pads_initializer { get_initializer(pads) };

        if (pads_initializer)
        {
            padding_value = to<axis_t>(pads_initializer.value());
        }
        else
        {
            // try to extract data from previous constant nodes
            const auto data { get_constant_input_data<float>(pads) };

            if (data)
                transform(begin(data.value()), end(data.value()), back_inserter(padding_value),
                    [](const auto e) { return static_cast<int>(e); });
        }
    }
    else
    {
        const auto padding_attr { get_attribute<axis_t>(node, "pads") };
        if (!padding_attr)
            throw runtime_error("\"pads\" attribute is required in Pad operator in opsets version 10 and lower");
        padding_value = padding_attr.value();
    }

    if (padding_value.size() < 4)
    {
        throw runtime_error("Only 2D padding is supported");
    }

    const xt::svector<padding>& new_paddings { parse_padding(padding_value) };

    scalar constant { (uint8_t)0 };

    if (!use_opset_version_9)
    {
        if (node.input().size() == 3)
        {
            const auto &constant_value { node.input()[2] };

            const auto &constant_initializer { get_initializer(constant_value) };
            switch (input_type)
            {
            case dt_float32:
            {
                if (constant_initializer)
                {
                    constant = to<float>(constant_initializer.value());
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
                    const auto data { get_constant_input_data<uint8_t>(constant_value) };

                    if (data && data.value().size())
                    {
                        constant = data.value()[0];
                    }
                }
                break;
            }
            }
        }
    }
    else
    {
        const auto constant_attr { get_attribute<float>(node, "value") };
        constant = constant_attr ? constant_attr.value() : 0;
    }

    if (constant.as<float>() != 0.0f)
        cout << "Warning: non-zero padding value specified, which is not supported by the hardware, it will be ignored" << endl;

    auto op { graph_.emplace<pad>(input_type, input_shape, new_paddings, move(constant)) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
