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

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Reshape(const NodeProto &node)
{
    const auto &op_name { generate_name(node) };

    const auto &input = node.input()[0];
    const auto &output = node.output()[0];

    const auto input_type = get_datatype(input).value();
    const auto &input_shape = get_shape(input);

    auto vec = get_constant_value<int, int64_t>(node.input()[1]);
    axis_t new_shape { vec.begin(), vec.end() };

    auto allowzero_attr = get_attribute<int>(node, "allowzero");
    int allowzero = !allowzero_attr ? 0 : allowzero_attr.value();

    const size_t size = new_shape.size();
    size_t negative_idx = size;

    // fixup dim which is zero
    for (size_t i = 0; i < size; i++)
    {
        if ((allowzero == 0) && (new_shape[i] == 0))
        {
            new_shape[i] = input_shape[i];
        }
        else if (new_shape[i] == -1)
        {
            negative_idx = i;
        }
    }

    // fixup dim which is -1
    if (negative_idx < size)
    {
        int product = 1;
        for (size_t i = 0; i < size; i++)
        {
            if (i == negative_idx)
                continue;
            product *= new_shape[i];
        }
        new_shape[negative_idx] = xt::compute_size(input_shape) / product;
    }

    auto op = graph_.emplace<bitcast>(input_type, input_shape, new_shape);
    op->name(op_name + "(Reshape)");

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
