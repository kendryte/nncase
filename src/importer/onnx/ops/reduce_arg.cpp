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
#include <nncase/ir/ops/reduce_arg.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_ArgMax(const NodeProto &node)
{
    convert_op_arg(node, reduce_arg_max);
}

void onnx_importer::convert_op_ArgMin(const NodeProto &node)
{
    convert_op_arg(node, reduce_arg_min);
}

void onnx_importer::convert_op_arg(const NodeProto &node, reduce_arg_op_t op)
{
    const auto &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    const auto &output = node.output()[0];
    const auto output_type = get_datatype(output).value();

    // axis
    auto axis_attr = get_attribute<int>(node, "axis");
    int32_t axis = axis_attr ? axis_attr.value() : 0;

    // keepdims
    auto keepdims_attr = get_attribute<int>(node, "keepdims");
    bool keepdims = keepdims_attr ? keepdims_attr.value() : true;

    // select_last_index
    auto select_last_index_attr = get_attribute<int>(node, "select_last_index");
    bool select_last_index = select_last_index_attr ? select_last_index_attr.value() : false;

    auto am = graph_.emplace<reduce_arg>(op, input_type, input_shape, output_type, axis, keepdims, select_last_index);
    am->name(generate_name(node) + reduce_arg_op_to_string(op));

    input_tensors_.emplace(&am->input(), input);
    output_tensors_.emplace(output, &am->output());
}
