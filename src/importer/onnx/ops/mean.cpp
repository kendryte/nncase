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
#include <nncase/importer/util.h>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/binary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Mean(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };
    std::vector<std::string> inputs;
    auto input_size = node.input().size();
    for (size_t i = 0; i < input_size; i++)
    {
        inputs.push_back(node.input()[i]);
    }
    const auto input_type = get_datatype(inputs[0]).value();
    auto output = node.output()[0];

    std::vector<ir::node *> adds;
    for (size_t i = 0; i < inputs.size() - 1; i++)
    {
        shape_t in_a_shape = i == 0 ? get_shape(inputs[i]) : adds[i - 1]->output_at(0).shape();
        shape_t in_b_shape = get_shape(inputs[i + 1]);
        auto add = graph_.emplace<binary>(binary_add, input_type, in_a_shape, in_b_shape, value_range<float>::full());
        add->name(op_name + "/add" + std::to_string(i));

        if (i == 0)
        {
            input_tensors_.emplace(&add->input_a(), inputs[0]);
            input_tensors_.emplace(&add->input_b(), inputs[1]);
        }
        else
        {
            add->input_a().connect(adds[i - 1]->output_at(0));
            input_tensors_.emplace(&add->input_b(), inputs[i + 1]);
        }
        adds.push_back(add);
    }

    auto n = graph_.emplace<constant>(3.f);
    n->name(op_name + ".n(Mean)");

    auto div = graph_.emplace<binary>(binary_div, input_type, adds.back()->output_at(0).shape(), n->output().shape(), value_range<float>::full());
    div->name(op_name + ".div(Mean)");

    div->input_a().connect(adds.back()->output_at(0));
    div->input_b().connect(n->output());

    output_tensors_.emplace(output, &div->output());
}
