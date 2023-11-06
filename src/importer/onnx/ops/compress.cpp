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
#include <nncase/ir/ops/compress.h>
#include <vector>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Compress(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    auto input = node.input()[0];
    auto condition = node.input()[1];
    auto output = node.output()[0];

    const auto in_type = get_datatype(input).value();
    const auto in_shape = get_shape(input);
    const auto condition_shape = get_shape(condition);
    const auto out_shape = get_shape(output);

    auto onnx_axis = get_attribute<int>(node, "axis").value_or((int)in_shape.size());

    auto onnx_compress = graph_.emplace<compress>(in_type, in_shape, condition_shape, out_shape, onnx_axis);
    onnx_compress->name(op_name);

    input_tensors_.emplace(&onnx_compress->input_at(0), node.input()[0]);
    input_tensors_.emplace(&onnx_compress->input_at(1), node.input()[1]);

    output_tensors_.emplace(node.output()[0], &onnx_compress->output());
}
