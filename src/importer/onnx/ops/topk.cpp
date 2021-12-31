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
#include <nncase/ir/ops/topk.h>
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_TopK(const NodeProto &node)
{
    // check opset version
    auto opset_version = get_opset_version();
    if (opset_version < 11)
    {
        throw std::runtime_error("opset less than 11 is not supported");
    }

    assert(node.output().size() == 2);
    const auto &op_name { generate_name(node) };

    // data input
    const auto &input = node.input()[0];
    const datatype_t input_type = get_datatype(input).value();
    auto input_shape = get_shape(input);

    // k input
    auto topk_vec = get_constant_value<int64_t>(node.input()[1]);
    assert(topk_vec.size() == 1 && topk_vec[0] > 0);

    // output
    const auto &output_values = node.output()[0];
    const auto &output_indices = node.output()[1];

    // axis
    int32_t axis = get_attribute<int32_t>(node, "axis").value_or(-1);

    // largest
    bool largest = get_attribute<int32_t>(node, "largest").value_or(1);

    // sorted
    bool sorted = get_attribute<int32_t>(node, "sorted").value_or(1);

    auto op = graph_.emplace<topk>(input_type, input_shape, topk_vec[0], axis, largest, sorted);
    op->name(op_name + "/topk");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output_values, &op->output_a());
    output_tensors_.emplace(output_indices, &op->output_b());
}
