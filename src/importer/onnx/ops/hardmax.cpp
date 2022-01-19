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
#include <nncase/ir/ops/hardmax.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Hardmax(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    const auto &output = node.output()[0];

    // axis
    auto axis_attr = get_attribute<int>(node, "axis");
    int32_t axis = axis_attr ? axis_attr.value() : -1;

    auto op = graph_.emplace<hardmax>(input_type, input_shape, axis);
    op->name(generate_name(node));

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
