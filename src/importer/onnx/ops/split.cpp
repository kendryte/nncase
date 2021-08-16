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
#include <nncase/ir/ops/split.h>
#include <vector>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Split(const NodeProto &node)
{
    auto input = node.input()[0];
    auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);

    // outputs shape
    auto output_size = node.output().size();
    std::vector<shape_t> outputs_shape(output_size);
    for (auto i = 0; i < output_size; i++)
    {
        outputs_shape[i] = get_shape(node.output()[i]);
    }

    // axis
    auto axis_attr = get_attribute<int>(node, "axis");
    int axis = axis_attr ? axis_attr.value() : 0;
    axis = real_axis(axis, input_shape.size());

    // split
    std::vector<int> splits;

    // opset 1/2/11
    auto split_attr = get_attribute<std::vector<int>>(node, "split");
    if (split_attr)
    {
        splits = split_attr.value();
    }

    // opset 1/13
    if (node.input().size() == 2)
    {
        splits = get_constant_value<int, int64_t>(node.input()[1]);
    }

    if (splits.empty())
    {
        assert(input_shape[axis] % output_size == 0);
        splits.assign(output_size, input_shape[axis] / output_size);
    }

    std::vector<size_t> indices(splits.begin(), splits.end());

    auto op = graph_.emplace<split>(input_type, input_shape, outputs_shape, indices, axis, true);
    op->name(generate_name(node) + ".(Split)");

    input_tensors_.emplace(&op->input(), input);
    for (size_t i = 0; i < output_size; i++)
    {
        output_tensors_.emplace(node.output()[i], &op->output_at(i));
    }
}
