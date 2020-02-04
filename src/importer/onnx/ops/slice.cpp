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
#include <hlir/ops/strided_slice.h>


using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Slice(const NodeProto& node)
{
    const string& data_input_name { node.input()[0] };
    const string& starts_input_name { node.input()[1] };
    const string& ends_input_name { node.input()[2] };

    const auto data_input_value_info { find_value_info(data_input_name) };

    if (!data_input_value_info)
        throw runtime_error("Can't find value info for data input for slice element");

    shape_t&& data_shape { get_shape(*data_input_value_info) };
    const datatype_t data_type { get_datatype(*data_input_value_info) };
    axis_t begins { to<axis_t>(get_initializer(starts_input_name)) };
    axis_t ends { to<axis_t>(get_initializer(ends_input_name)) };

    assert(begins.size() == ends.size());
    assert(begins.size() <= data_shape.size());

    axis_t axes(begins.size()), strides(begins.size());
    iota(begin(axes), end(axes), 0);
    fill(begin(strides), end(strides), 1);

    if (node.input().size() > 3)
    {
        const string& axes_input_name { node.input()[3] };
        const auto &loaded_axes { to<axis_t>(get_initializer(axes_input_name)) };

        if (!equal(begin(loaded_axes), end(loaded_axes), begin(axes), end(axes)))
        {
            cout << "Warning: non-sequential values in 'axes' input of slice element is not supported and ignored, axes are meant to be sequential" << endl;
        }
    }

    if (node.input().size() > 4)
    {
        const string& strides_input_name { node.input()[4] };
        strides = to<axis_t>(get_initializer(strides_input_name));
        assert(strides.size() == begins.size());
    }

    auto sl { graph_.emplace<strided_slice>(data_type, data_shape, begins, ends, strides, 0, 0, 0, 0, 0) };

    input_tensors_.emplace(&sl->input(), data_input_name);
    output_tensors_.emplace(node.output()[0], &sl->output());
}
