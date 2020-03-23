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

#include <vector>
#include <cassert>

#include <hlir/graph.h>
#include <hlir/ops/concat.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

namespace
{
    datatype_t deduce_common_type(const vector<datatype_t>& datatypes) noexcept
    {
        assert(!datatypes.empty());
        assert(dt_float32 < dt_uint8);
        return *min_element(begin(datatypes), end(datatypes));
    }

    size_t axis_count(const vector<shape_t>& shapes)
    {
        if (shapes.empty())
        {
            return 0;
        }

        return shapes.front().size();
    }
}

void onnx_importer::convert_op_Concat(const NodeProto &node)
{
    vector<shape_t> inputs_shapes;
    vector<datatype_t> inputs_types;

    for (const string& input_name : node.input())
    {
        inputs_shapes.push_back(get_shape(input_name));
        inputs_types.push_back(get_datatype(input_name).value());
    }

    const datatype_t op_type { deduce_common_type(inputs_types) };
    const size_t axis { real_axis(get_attribute<int64_t>(node, "axis").value(), axis_count(inputs_shapes)) };

    auto con { graph_.emplace<concat>(op_type, inputs_shapes, axis) };

    for (size_t i = 0; i < node.input().size(); i++)
        input_tensors_.emplace(&con->input_at(i), node.input()[i]);

    output_tensors_.emplace(node.output()[0], &con->output());
}
