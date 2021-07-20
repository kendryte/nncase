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
#include <algorithm>
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/bitcast.h>
#include <numeric>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

namespace
{
axis_t compose_new_shape(const shape_t &input_shape, const size_t flatten_axis)
{
    axis_t result;
    std::copy(std::begin(input_shape), std::begin(input_shape) + flatten_axis, std::back_inserter(result));
    const auto flattened_axis = std::accumulate(std::begin(input_shape) + flatten_axis, std::end(input_shape), 1, std::multiplies<long>());
    result.push_back(flattened_axis);

    return result;
}
}

void onnx_importer::convert_op_Flatten(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    const auto &axis_attr = get_attribute<int>(node, "axis");
    const size_t flatten_axis = axis_attr ? real_axis(axis_attr.value(), input_shape.size()) : 1;
    const axis_t &new_shape = compose_new_shape(input_shape, flatten_axis);
    auto op = graph_.emplace<bitcast>(input_type, input_shape, new_shape);
    op->name(op_name + "(Flatten)");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
