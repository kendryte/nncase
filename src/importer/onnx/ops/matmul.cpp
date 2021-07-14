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
    const auto &output = node.output()[0];

    auto &&input_a_shape = get_shape(input_a);
    auto &&input_b_shape = get_shape(input_b);

    std::vector<float> bias_value(input_b_shape.back(), 0.f);
    shape_t bias_shape = { input_b_shape.back() };
    auto bias = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
    bias->name(op_name + ".bias(MatMul)");

    auto mmul = graph_.emplace<matmul>(std::move(input_a_shape), std::move(input_b_shape), value_range<float>::full());
    mmul->name(op_name + ".matmul(MatMul)");
    mmul->bias().connect(bias->output());

    input_tensors_.emplace(&mmul->input_a(), input_a);
    input_tensors_.emplace(&mmul->input_b(), input_b);
    output_tensors_.emplace(output, &mmul->output());
}
