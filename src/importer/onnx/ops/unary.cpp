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
#include <hlir/ops/constant.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/binary.h>
#include <hlir/ops/unary.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Abs(const onnx::NodeProto &node)
{
    convert_unary(node, unary_abs);
}

void onnx_importer::convert_op_Ceil(const onnx::NodeProto &node)
{
    convert_unary(node, unary_ceil);
}

void onnx_importer::convert_op_Floor(const onnx::NodeProto &node)
{
    convert_unary(node, unary_floor);
}

void onnx_importer::convert_op_Cos(const onnx::NodeProto &node)
{
    convert_unary(node, unary_cos);
}

void onnx_importer::convert_op_Sin(const onnx::NodeProto &node)
{
    convert_unary(node, unary_sin);
}

void onnx_importer::convert_op_Exp(const onnx::NodeProto &node)
{
    convert_unary(node, unary_exp);
}

void onnx_importer::convert_op_Log(const onnx::NodeProto &node)
{
    convert_unary(node, unary_log);
}

void onnx_importer::convert_op_Neg(const onnx::NodeProto &node)
{
    convert_unary(node, unary_neg);
}

void onnx_importer::convert_op_Sqrt(const onnx::NodeProto &node)
{
    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const auto input_datatype { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    auto op { graph_.emplace<unary>(unary_rsqrt, input_shape) };

    hlir::constant* one { };
    switch (input_datatype)
    {
    default:
    case dt_float32:
        one = graph_.emplace<constant>(float(1));
        break;

    case dt_uint8:
        one = graph_.emplace<constant>(uint8_t(1));
        break;
    }

    auto dv { graph_.emplace<binary>(binary_div, one->output().shape(), op->output().shape(), value_range<float>::full()) };

    dv->input_a().connect(one->output());
    dv->input_b().connect(op->output());

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &dv->output());
}

void onnx_importer::convert_unary(const onnx::NodeProto &node, const unary_op_t unary_op)
{

    assert(node.input().size() == 1);
    assert(node.output().size() == 1);

    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const auto &input_shape { get_shape(input) };

    auto op { graph_.emplace<unary>(unary_op, input_shape) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
