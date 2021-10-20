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
#include <nncase/ir/ops/cumsum.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_CumSum(const NodeProto &node)
{
    assert(node.input().size() >= 2);
    assert(node.output().size() == 1);

    const auto &input = node.input()[0];
    const auto input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);
    const auto &output = node.output()[0];
    // const auto output_type = get_datatype(output).value();

    // exclusive
    auto exclusive_attr = get_attribute<int>(node, "exclusive");
    bool exclusive = exclusive_attr ? exclusive_attr.value() : false;

    // reverse
    auto reverse_attr = get_attribute<int>(node, "reverse");
    bool reverse = reverse_attr ? reverse_attr.value() : false;

    // axis
    auto axes = get_constant_value<int32_t>(node.input()[1]);
    assert(axes.size() == 1);

    auto op = graph_.emplace<cumsum>(input_type, input_shape, axes[0], exclusive, reverse);
    op->name(generate_name(node));

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
