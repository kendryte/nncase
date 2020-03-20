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

#include <hlir/graph.h>
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/unary.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Relu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    auto&& in_shape = get_shape(input);

    auto zero { graph_.emplace<constant>(0.f) };
    auto max { graph_.emplace<binary>(binary_max, move(in_shape), zero->output().shape(), value_range<float>::full()) };

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}

void onnx_importer::convert_op_LeakyRelu(const NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };
    auto&& in_shape = get_shape(input);

    const auto alpha_value { get_attribute<float>(node, "alpha").value() };
    const auto& alpha { graph_.emplace<constant>(get_datatype<float>(), alpha_value) };

    auto mul = graph_.emplace<binary>(binary_mul, move(in_shape), alpha->output().shape(), value_range<float>::full());
    auto max = graph_.emplace<binary>(binary_max, move(in_shape), mul->output().shape(), value_range<float>::full());

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&max->input_a(), input);
    output_tensors_.emplace(output, &max->output());
}
