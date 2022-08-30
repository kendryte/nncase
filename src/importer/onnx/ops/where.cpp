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
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/ternary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Where(const onnx::NodeProto &node)
{
    assert(node.input().size() == 3);
    assert(node.output().size() == 1);

    const auto &op_name { generate_name(node) };
    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto &input_c = node.input()[2];
    const auto &output = node.output()[0];

    datatype_t dtype = dt_float32;
    auto deq_a = graph_.emplace<convert>(get_datatype(input_a).value(), get_shape(input_a), dtype);
    deq_a->name(op_name + "/cvt");

    auto op = graph_.emplace<ternary>(dtype, get_datatype(input_b).value(), deq_a->output().shape(), get_shape(input_b), get_shape(input_c));
    op->name(op_name + "/ternary");
    op->input_a().connect(deq_a->output());

    input_tensors_.emplace(&deq_a->input(), input_a);
    input_tensors_.emplace(&op->input_b(), input_b);
    input_tensors_.emplace(&op->input_c(), input_c);
    output_tensors_.emplace(output, &op->output());
}
