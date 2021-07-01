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

void onnx_importer::convert_op_Dropout(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const datatype_t input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    // Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input
    auto bc = graph_.emplace<bitcast>(input_type, input_shape, input_shape);
    input_tensors_.emplace(&bc->input(), input);
    output_tensors_.emplace(output, &bc->output());
}