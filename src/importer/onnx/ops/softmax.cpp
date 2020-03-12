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


using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Softmax(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    auto input_shape { get_shape(input) };

    axis_t reduce_axis { static_cast<int>(real_axis(static_cast<int>(get_attribute<int64_t>(node, "axis").value()), input_shape.size())) };

    auto max { graph_.emplace<reduce>(reduce_max, input_shape, reduce_axis, std::numeric_limits<float>::lowest(), true) };
    auto sub { graph_.emplace<binary>(binary_sub, input_shape, max->output().shape(), value_range<float>::full()) };
    auto exp { graph_.emplace<unary>(unary_exp, sub->output().shape()) };
    auto sum { graph_.emplace<reduce>(reduce_sum, exp->output().shape(), reduce_axis, 0.f, true) };
    auto div { graph_.emplace<binary>(binary_div, exp->output().shape(), sum->output().shape(), value_range<float>::full()) };

    sub->input_b().connect(max->output());
    exp->input().connect(sub->output());
    sum->input().connect(exp->output());
    div->input_a().connect(exp->output());
    div->input_b().connect(sum->output());

    input_tensors_.emplace(&max->input(), input);
    input_tensors_.emplace(&sub->input_a(), input);
    output_tensors_.emplace(output, &div->output());
}
