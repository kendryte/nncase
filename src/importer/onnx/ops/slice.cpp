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
    const string &data_input_name { node.input()[0] };

    const datatype_t data_type { get_datatype(data_input_name).value() };
    const shape_t &data_shape { get_shape(data_input_name) };

    axis_t begins, ends;

    const bool use_opset_version_9 { node.input().size() == 1 };

    if (!use_opset_version_9)
    {
        const string& starts_input_name { node.input()[1] };
        const string& ends_input_name { node.input()[2] };

        const auto &begins_initializer { get_initializer(starts_input_name) };
        const auto &ends_initializer{ get_initializer(ends_input_name) };

        if (!begins_initializer)
        {
            // try to extract data from previous constant nodes
            const auto data { get_constant_input_data<float>(starts_input_name) };

            if (!data)
                throw runtime_error("Can't pull input data for slice starts: only constant initialization is supported");

            transform(begin(data.value()), end(data.value()), back_inserter(begins),
                [](const auto e) { return static_cast<int>(e); });
        }
        else
        {
            begins = to<axis_t>(begins_initializer.value());
        }

        if (!ends_initializer)
        {
            // try to extract data from previous constant nodes
            const auto data { get_constant_input_data<float>(ends_input_name) };

            if (!data)
                throw runtime_error("Can't pull input data for slice ends: only constant initialization is supported");

            transform(begin(data.value()), end(data.value()), back_inserter(ends),
                [](const auto e) { return static_cast<int>(e); });
        }
        else
        {
            ends = to<axis_t>(ends_initializer.value());
        }
    }
    else
    {
        begins = get_attribute<axis_t>(node, "starts").value();
        ends = get_attribute<axis_t>(node, "ends").value();
    }

    assert(begins.size() == ends.size());
    assert(begins.size() <= data_shape.size());

    axis_t axes(data_shape.size()), strides(data_shape.size());
    iota(begin(axes), end(axes), 0);
    fill(begin(strides), end(strides), 1);

    axis_t loaded_axes, loaded_strides;

    if (!use_opset_version_9)
    {
        if (node.input().size() > 3)
        {
            const string &axes_input_name { node.input()[3] };
            const auto &axes_initializer{ get_initializer(axes_input_name) };

            if (!axes_initializer)
            {
                // try to extract data from previous constant nodes
                const auto data { get_constant_input_data<float>(axes_input_name) };

                if (data)
                    transform(begin(data.value()), end(data.value()), back_inserter(loaded_axes),
                        [](const auto e) { return static_cast<int>(e); });
            }
            else
            {
                loaded_axes = to<axis_t>(axes_initializer.value());
            }
        }
    }
    else
    {
        const auto axes_attr { get_attribute<axis_t>(node, "axes") };

        if (axes_attr)
            loaded_axes = axes_attr.value();
    }

    if (node.input().size() > 4)
    {
        const string &strides_input_name { node.input()[4] };

        const auto &strides_initializer{ get_initializer(strides_input_name) };

        if (!strides_initializer)
        {
            // try to extract data from previous constant nodes
            const auto data { get_constant_input_data<float>(strides_input_name) };

            if (data)
                transform(begin(data.value()), end(data.value()), back_inserter(loaded_strides),
                    [](const auto e) { return static_cast<int>(e); });
        }
        else
        {
            loaded_strides = to<axis_t>(strides_initializer.value());
        }

        assert(loaded_strides.size() == loaded_axes.size());
    }

    axis_t permuted_begins, permuted_ends;
    for (size_t i = 0; i < axes.size(); ++i)
    {
        const auto it { find_if(begin(loaded_axes), end(loaded_axes), [i, &data_shape](const auto e) { return real_axis(e, data_shape.size()) == i; }) };

        if (it == end(loaded_axes))
        {
            permuted_begins.push_back(0);
            permuted_ends.push_back(data_shape[i]);
        }
        else
        {
            const size_t index { static_cast<size_t>(it - begin(loaded_axes)) };
            permuted_begins.push_back(begins.at(index));
            permuted_ends.push_back(ends.at(index));

            if (!loaded_strides.empty())
                strides[i] = loaded_strides.at(index);
        }
    }

    begins = permuted_begins;
    ends = permuted_ends;

    auto sl { graph_.emplace<strided_slice>(data_type, data_shape, begins, ends, strides, 0, 0, 0, 0, 0) };

    input_tensors_.emplace(&sl->input(), data_input_name);
    output_tensors_.emplace(node.output()[0], &sl->output());
}
