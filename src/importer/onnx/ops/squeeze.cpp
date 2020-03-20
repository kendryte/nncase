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
#include <hlir/ops/reshape.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    axis_t single_dim_axes(const shape_t& shape)
    {
        axis_t result;

        for (auto i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 1)
                result.push_back(i);
        }

        return result;
    }
}

void onnx_importer::convert_op_Squeeze(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &output { node.output()[0] };

    const auto input_type { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    const auto axes_attr { get_attribute<axis_t>(node, "axes") };

    auto new_shape { input_shape };
    const axis_t axes { axes_attr ? axes_attr.value() : single_dim_axes(input_shape) };

    size_t squeezed_count { };
    for (const auto axis : axes)
    {
        const auto fixed_axis { real_axis(axis, input_shape.size()) };
        if (input_shape.at(fixed_axis) != 1)
            throw runtime_error("Only single-dimensional axes can be squeezed");

        new_shape.erase(begin(new_shape) + fixed_axis - squeezed_count);
        ++squeezed_count;
    }

    auto op { graph_.emplace<reshape>(input_type, input_shape, new_shape) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
