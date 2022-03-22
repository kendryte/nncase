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
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/slice.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_MatMul(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    // input A
    const auto &input_a = node.input()[0];
    const auto input_type = get_datatype(input_a).value();
    const auto &input_a_shape = get_shape(input_a);
    assert(input_a_shape.size() >= 1);

    // input B
    const auto &input_b = node.input()[1];
    const auto &input_b_shape = get_shape(input_b);
    assert(input_b_shape.size() >= 1);

    // output
    const auto &output = node.output()[0];
    const auto &output_shape = get_shape(output);

    // reshape A to [batch, n, k]
    shape_t new_a_shape { 1, 1, 1 };
    auto input_a_shape_size = input_a_shape.size();
    if (input_a_shape_size == 1)
    {
        new_a_shape[2] = input_a_shape[0];
    }
    else if (input_a_shape_size == 2)
    {
        new_a_shape[1] = input_a_shape[0];
        new_a_shape[2] = input_a_shape[1];
    }
    else if (input_a_shape_size == 3)
    {
        new_a_shape.assign(input_a_shape.begin(), input_a_shape.end());
    }
    else
    {
        for (size_t i = 0; i < input_a_shape_size - 2; i++)
        {
            new_a_shape[0] *= input_a_shape[i];
        }

        new_a_shape[1] = input_a_shape[input_a_shape_size - 2];
        new_a_shape[2] = input_a_shape[input_a_shape_size - 1];
    }

    auto bc_a_3d = graph_.emplace<bitcast>(input_type, input_a_shape, new_a_shape);
    bc_a_3d->name(op_name + ".bitcast_A_3d(MatMul)");

    // reshape B to [batch, k, m]
    shape_t new_b_shape = { 1, 1, 1 };
    auto input_b_shape_size = input_b_shape.size();
    if (input_b_shape_size == 1)
    {
        new_b_shape[1] = input_b_shape[0];
    }
    else if (input_b_shape_size == 2)
    {
        new_b_shape[1] = input_b_shape[0];
        new_b_shape[2] = input_b_shape[1];
    }
    else if (input_b_shape_size == 3)
    {
        new_b_shape.assign(input_b_shape.begin(), input_b_shape.end());
    }
    else
    {
        for (size_t i = 0; i < input_b_shape_size - 2; i++)
        {
            new_b_shape[0] *= input_b_shape[i];
        }

        new_b_shape[1] = input_b_shape[input_b_shape_size - 2];
        new_b_shape[2] = input_b_shape[input_b_shape_size - 1];
    }

    auto bc_b_3d = graph_.emplace<bitcast>(input_type, input_b_shape, new_b_shape);
    bc_b_3d->name(op_name + ".bitcast_B_3d(MatMul)");

    assert(new_a_shape[2] == new_b_shape[1]);
    if (!((new_a_shape[0] == 1) || (new_b_shape[0] == 1) || (new_a_shape[0] == new_b_shape[0])))
    {
        throw std::runtime_error("we don't support such broadcast for matmul.");
    }

    // output shape
    shape_t new_output_shape { 1, new_a_shape[1], new_b_shape[2] };
    new_output_shape[0] = new_a_shape[0] == 1 ? new_b_shape[0] : new_a_shape[0];

    bitcast *bc_a = bc_a_3d;
    if (new_a_shape[0] == 1)
    {
        // reshape to 2D
        shape_t new_shape { new_a_shape[1], new_a_shape[2] };
        bc_a = graph_.emplace<bitcast>(input_type, bc_a_3d->output().shape(), new_shape);
        bc_a->name(op_name + ".bitcast_A(MatMul)");
        bc_a->input().connect(bc_a_3d->output());
    }

    bitcast *bc_b = bc_b_3d;
    if (new_b_shape[0] == 1)
    {
        // reshape to 2D
        shape_t new_shape { new_b_shape[1], new_b_shape[2] };
        bc_b = graph_.emplace<bitcast>(input_type, bc_b_3d->output().shape(), new_shape);
        bc_b->name(op_name + ".bitcast_B(MatMul)");
        bc_b->input().connect(bc_b_3d->output());
    }

    // bias
    auto b = bc_b->output().shape().back();
    std::vector<float> bias_value(b, 0.f);
    shape_t bias_shape = { b };
    auto bias = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
    bias->name(op_name + ".bias(MatMul)");

    // matmul
    auto mm = graph_.emplace<matmul>(bc_a->output().shape(), bc_b->output().shape(), value_range<float>::full());
    mm->name(op_name + ".matmul(MatMul)");
    mm->input_a().connect(bc_a->output());
    mm->input_b().connect(bc_b->output());
    mm->bias().connect(bias->output());

    // reshape to output
    auto bc_output = graph_.emplace<bitcast>(input_type, mm->output().shape(), output_shape);
    bc_output->name(op_name + ".bitcast(MatMul)");
    bc_output->input().connect(mm->output());

    input_tensors_.emplace(&bc_a_3d->input(), input_a);
    input_tensors_.emplace(&bc_b_3d->input(), input_b);
    output_tensors_.emplace(output, &bc_output->output());
}
