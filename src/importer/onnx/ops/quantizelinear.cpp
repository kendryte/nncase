/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/ir/ops/quantize.h>
#include <nncase/runtime/debug.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_QuantizeLinear(const onnx::NodeProto &node)
{
    assert(node.input().size() >= 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &scale = node.input()[1];
    const auto &output = node.output()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    const auto output_type = get_datatype(output).value();

    // Ignored for per-tensor quantization.
    // auto axis_attr = get_attribute<int>(node, "axis");
    // auto axis = axis_attr ? axis_attr.value() : 1;
    // axis = normalize_axis(input_shape, axis);

    quant_param_t qparam;

    // scale
    auto scale_value = get_constant_value<float>(scale);
    if (scale_value.size() > 1)
    {
        throw std::runtime_error("scale: only support per-tensor quantization.");
    }
    qparam.scale = scale_value[0];

    // zero point
    if (node.input().size() == 2)
    {
        qparam.zero_point = 0;
    }
    else
    {
        auto zp = node.input()[2];
        switch (output_type)
        {
        case dt_uint8:
        {
            auto zp_value = get_constant_value<uint8_t>(zp);
            if (zp_value.size() > 1)
            {
                throw std::runtime_error("zero point: only support per-tensor quantization.");
            }
            qparam.zero_point = zp_value[0];
            break;
        }

        case dt_int8:
        {
            auto zp_value = get_constant_value<int8_t>(zp);
            if (zp_value.size() > 1)
            {
                throw std::runtime_error("zero point: only support per-tensor quantization.");
            }
            qparam.zero_point = zp_value[0];
            break;
        }
        default:
        {
            throw std::runtime_error("QuantizeLinear: unsupported data type:" + std::string(datatype_names(output_type)));
        }
        }
    }

    auto op = graph_.emplace<quantize>(input_type, input_shape, output_type, qparam);
    op->name(op_name);

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
