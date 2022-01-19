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
#include <nncase/ir/ops/equal.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Equal(const onnx::NodeProto &node)
{
    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    auto input_a_type = get_datatype(input_a).value();
    auto input_a_shape = get_shape(input_a);

    const auto &input_b = node.input()[1];
    auto input_b_shape = get_shape(input_b);

    const auto &output = node.output()[0];

    auto op = graph_.emplace<equal>(input_a_type, input_a_shape, input_b_shape);
    op->name(op_name + "/equal");

    input_tensors_.emplace(&op->input_a(), input_a);
    input_tensors_.emplace(&op->input_b(), input_b);
    output_tensors_.emplace(output, &op->output());
}
