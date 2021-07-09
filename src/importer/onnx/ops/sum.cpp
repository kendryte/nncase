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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Sum(const onnx::NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    assert(node.input().size() >= 1);
    assert(node.output().size() == 1);

    const auto &output = node.output()[0];
    binary *last_op = nullptr;

    auto size = node.input().size();
    for (int i = 0; i < size - 1; i++)
    {
        auto input_a = node.input()[i];
        auto input_b = node.input()[i + 1];

        auto input_a_shape = get_shape(input_a);
        auto input_b_shape = get_shape(input_b);

        auto cur_op = graph_.emplace<binary>(binary_add, input_a_shape, input_b_shape, value_range<float>::full());
        cur_op->name(op_name + ".add(Sum)");
        if (last_op != nullptr)
        {
            cur_op->input_a().connect(last_op->output());
        }
        else
        {
            input_tensors_.emplace(&cur_op->input_a(), input_a);
        }
        input_tensors_.emplace(&cur_op->input_b(), input_b);
        last_op = cur_op;
    }

    if (last_op != nullptr)
    {
        // more than one input
        output_tensors_.emplace(output, &last_op->output());
    }
    else
    {
        // only one input
        auto input = node.input()[0];
        const datatype_t input_type = get_datatype(input).value();
        const auto &input_shape = get_shape(input);
        auto bc = graph_.emplace<bitcast>(input_type, input_shape, input_shape);
        bc->name(op_name + ".broadcast(Sum)");
        input_tensors_.emplace(&bc->input(), input);
        output_tensors_.emplace(output, &bc->output());
    }
}
