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

void onnx_importer::convert_op_Add(const onnx::NodeProto &node)
{
    convert_binary(node, binary_add);
}

void onnx_importer::convert_op_Sub(const onnx::NodeProto &node)
{
    convert_binary(node, binary_sub);
}

void onnx_importer::convert_op_Mul(const onnx::NodeProto &node)
{
    convert_binary(node, binary_mul);
}

void onnx_importer::convert_op_Div(const onnx::NodeProto &node)
{
    convert_binary(node, binary_div);
}

void onnx_importer::convert_op_Min(const onnx::NodeProto &node)
{
    convert_binary(node, binary_min);
}

void onnx_importer::convert_op_Max(const onnx::NodeProto &node)
{
    convert_binary(node, binary_max);
}

void onnx_importer::convert_binary(const onnx::NodeProto &node, const binary_op_t binary_op)
{

    assert(node.input().size() == 2);
    assert(node.output().size() == 1);

    const auto &input_a { node.input()[0] }, &input_b { node.input()[1] };
    const auto &output { node.output()[0] };

    auto &&input_a_shape { get_shape(input_a) }, &&input_b_shape { get_shape(input_b) };

    auto op { graph_.emplace<binary>(binary_op, input_a_shape, input_b_shape, value_range<float>::full()) };

    input_tensors_.emplace(&op->input_a(), input_a);
    input_tensors_.emplace(&op->input_b(), input_b);
    output_tensors_.emplace(output, &op->output());
}
