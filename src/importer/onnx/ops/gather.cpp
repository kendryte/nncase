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
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/gather.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Gather(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &indices = node.input()[1];
    const auto &output = node.output()[0];

    const datatype_t input_type = get_datatype(input).value();
    const auto input_shape = get_shape(input);
    const auto indices_shape = get_shape(indices);
    const auto out_shape = get_shape(output);

    int32_t axis = 0;
    const auto axis_attr = get_attribute<int32_t>(node, "axis");
    if (axis_attr)
    {
        axis = static_cast<int32_t>(axis_attr.value());
    }
    if (axis < 0)
    {
        axis = static_cast<int32_t>(input_shape.size()) + axis;
    }

    auto ga = graph_.emplace<gather>(input_type, input_shape, indices_shape, out_shape, axis);
    if (input_type != dt_int32)
    {
        auto ct = graph_.emplace<convert>(get_datatype(indices).value(), indices_shape, dt_int32);
        ga->indices().connect(ct->output());
        input_tensors_.emplace(&ga->input(), input);
        input_tensors_.emplace(&ct->input(), indices);
        output_tensors_.emplace(output, &ga->output());
    }
    else
    {
        input_tensors_.emplace(&ga->input(), input);
        input_tensors_.emplace(&ga->indices(), indices);
        output_tensors_.emplace(output, &ga->output());
    }
}