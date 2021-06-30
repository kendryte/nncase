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
#include <nncase/ir/ops/bitcast.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Unsqueeze(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto input_shape = get_shape(input);

    axis_t axes;

    // Unsqueeze-1 and Unsqueeze-11 use axes as attribute
    const auto axes_attr = get_attribute<axis_t>(node, "axes");
    if (axes_attr)
    {
        axes = axes_attr.value();
    }
    else
    {
        // Unsqueeze-13 use axes as input[1]
        assert(node.input().size() == 2);
        auto axes_input = node.input()[1];
        auto initializer = get_initializer(axes_input);
        axes = initializer ? to<axis_t>(initializer.value()) : get_constant_input_data<int>(axes_input).value();
    }

    size_t size = input_shape.size() + axes.size();
    for (auto &axis : axes)
    {
        axis = real_axis(axis, size);
    }

    shape_t new_shape(size, 1);
    for (size_t i = 0, j = 0; i < size; i++)
    {
        new_shape[i] = std::find(axes.begin(), axes.end(), i) == axes.end() ?
                       input_shape[j++] : 1;
    }

    auto op = graph_.emplace<bitcast>(input_type, input_shape, new_shape);

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}