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
}

void onnx_importer::convert_op_Concat(const NodeProto &node)
{
    vector<shape_t> inputs_shapes;
    vector<datatype_t> inputs_types;

    for (const string& input_name : node.input())
    {
        const auto value_info_ptr { find_value_info(input_name) };

        if (value_info_ptr)
        {
            inputs_shapes.push_back(get_shape(*value_info_ptr));
            inputs_types.push_back(get_datatype(*value_info_ptr));
        }
        else
        {
            throw runtime_error("Can't find value info for " + input_name + " input");
        }
    }

    const datatype_t op_type { deduce_common_type(inputs_types) };
    const size_t axis { static_cast<size_t>(get_attribute<int64_t>(node, "axis").value()) };

    auto con { graph_.emplace<concat>(op_type, inputs_shapes, axis) };

    for (size_t i = 0; i < node.input().size(); i++)
        input_tensors_.emplace(&con->input_at(i), node.input()[i]);

    output_tensors_.emplace(node.output()[0], &con->output());
}
