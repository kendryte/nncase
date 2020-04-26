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

void onnx_importer::convert_op_Reshape(const NodeProto& node)
{
    const auto &input { node.input()[0] };
    const auto &shape { node.input()[1] };
    const auto &output { node.output()[0] };

    const auto input_type { get_datatype(input).value() };
    const auto &input_shape { get_shape(input) };

    const auto &new_shape_initializer { get_initializer(shape) };

    axis_t new_shape;

    if (new_shape_initializer)
    {
        new_shape = to<axis_t>(new_shape_initializer.value());
    }
    else
    {
        // try to extract data from previous constant nodes
        const auto data { get_constant_input_data<float>(shape) };

        if (data)
            transform(begin(data.value()), end(data.value()), back_inserter(new_shape),
                [](const auto e) { return static_cast<int>(e); });
    }

    auto op { graph_.emplace<reshape>(input_type, input_shape, new_shape) };

    input_tensors_.emplace(&op->input(), input);
    output_tensors_.emplace(output, &op->output());
}
