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
#include <limits>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Clip(const NodeProto &node)
{
    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    constant *min_op = nullptr;
    std::string input_min;
    if (node.input().size() < 2)
    {
        const auto min_attr = get_attribute<float>(node, "min");
        min_op = graph_.emplace<constant>(min_attr ? min_attr.value() : std::numeric_limits<float>::lowest());
    }
    else
    {
        input_min = node.input()[1];
    }

    constant *max_op = nullptr;
    std::string input_max;
    if (node.input().size() < 3)
    {
        const auto max_attr = get_attribute<float>(node, "max");
        max_op = graph_.emplace<constant>(max_attr ? max_attr.value() : std::numeric_limits<float>::max());
    }
    else
    {
        input_max = node.input()[2];
    }

    auto op = graph_.emplace<clamp>(get_shape(input),
        min_op ? min_op->output().shape() : get_shape(input_min),
        max_op ? max_op->output().shape() : get_shape(input_max));

    input_tensors_.emplace(&op->input(), input);

    if (min_op)
        op->input_low().connect(min_op->output());
    else
        input_tensors_.emplace(&op->input_low(), input_min);

    if (max_op)
        op->input_high().connect(max_op->output());
    else
        input_tensors_.emplace(&op->input_high(), input_max);

    output_tensors_.emplace(output, &op->output());
}