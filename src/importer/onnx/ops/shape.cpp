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
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Shape(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];
    const auto &input_shape = get_shape(input);

    std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
    auto size = shape.size();

    // optional start attribute for opset 15
    const auto start_attr = get_attribute<int>(node, "start");
    int start = start_attr ? start_attr.value() : 0;
    start = real_axis(start, size);

    // optional end attribute for opset 15
    const auto end_attr = get_attribute<int>(node, "end");
    int end = end_attr ? end_attr.value() : size;
    end = real_axis(end, size);

    std::vector<int64_t> slice(shape.begin() + start, shape.begin() + end);
    auto op = graph_.emplace<constant>(dt_int64, shape_t { slice.size() }, slice);
    op->name(generate_name(node) + "(Shape)");

    output_tensors_.emplace(output, &op->output());
}