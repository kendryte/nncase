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
#include <nncase/ir/ops/matmul.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_MatMul(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input_a = node.input()[0];
    const auto &input_b = node.input()[1];
    const auto input_type = get_datatype(input_a).value();
    const auto &output = node.output()[0];

    auto &&input_a_shape = get_shape(input_a);
    auto &&input_b_shape = get_shape(input_b);

    assert(input_a_shape.size() >= 1);
    assert(input_b_shape.size() >= 1);
    if (input_b_shape.size() > 2)
    {
        throw std::runtime_error("We don't support B's shape size > 2");
    }

    // output shape
    shape_t new_output_shape;

    // reshape A
    shape_t new_a_shape;
    if (input_a_shape.size() == 1)
    {
        new_a_shape.push_back(1);
        new_a_shape.push_back(input_a_shape[0]);
    }
    else if (input_a_shape.size() == 2)
    {
        new_a_shape.assign(input_a_shape.begin(), input_a_shape.end());
        new_output_shape.push_back(input_a_shape[0]);
    }
    else
    {
        new_a_shape.push_back(1);
        for (size_t i = 0; i < input_a_shape.size() - 1; i++)
        {
            new_a_shape[0] *= input_a_shape[i];
            new_output_shape.push_back(input_a_shape[i]);
        }
        new_a_shape.push_back(input_a_shape.back());
    }

    auto bc_a = graph_.emplace<bitcast>(input_type, input_a_shape, new_a_shape);
    bc_a->name(op_name + ".bitcast_A(MatMul)");

    // reshape B
    shape_t new_b_shape;
    if (input_b_shape.size() == 1)
    {
        new_b_shape.push_back(input_b_shape[0]);
        new_b_shape.push_back(1);
    }
    else if (input_b_shape.size() == 2)
    {
        new_b_shape.assign(input_b_shape.begin(), input_b_shape.end());
        new_output_shape.push_back(input_b_shape[1]);
    }
    else
    {
        throw std::runtime_error("We don't support B's shape size > 2");
    }

    if (new_output_shape.empty())
    {
        new_output_shape.push_back(1);
    }

    auto bc_b = graph_.emplace<bitcast>(input_type, input_b_shape, new_b_shape);
    bc_b->name(op_name + ".bitcast_B(MatMul)");

    std::vector<float> bias_value(new_b_shape.back(), 0.f);
    shape_t bias_shape = { new_b_shape.back() };
    auto bias = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
    bias->name(op_name + ".bias(MatMul)");

    auto mmul = graph_.emplace<matmul>(new_a_shape, new_b_shape, value_range<float>::full());
    mmul->name(op_name + ".matmul(MatMul)");
    mmul->input_a().connect(bc_a->output());
    mmul->input_b().connect(bc_b->output());
    mmul->bias().connect(bias->output());

    auto bc_output = graph_.emplace<bitcast>(input_type, mmul->output().shape(), new_output_shape);
    bc_output->name(op_name + ".bitcast_output(MatMul)");
    bc_output->input().connect(mmul->output());

    input_tensors_.emplace(&bc_a->input(), input_a);
    input_tensors_.emplace(&bc_b->input(), input_b);
    output_tensors_.emplace(output, &bc_output->output());
}
