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
#include <nncase/ir/ops/slice.h>
#include <nncase/runtime/datatypes.h>
#include <vector>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

void onnx_importer::convert_op_Slice(const NodeProto &node)
{
    const std::string &input = node.input()[0];
    const datatype_t input_type = get_datatype(input).value();
    const shape_t &input_shape = get_shape(input);
    auto ndim = input_shape.size();

#define GET_ATTRIBUTE(index, dst)                                             \
    {                                                                         \
        const std::string &name = node.input()[index];                        \
        const datatype_t type = get_datatype(name).value();                   \
                                                                              \
        if (type == datatype_t::dt_int32)                                     \
        {                                                                     \
            auto vec = get_constant_value<int, int32_t>(node.input()[index]); \
            dst.assign(vec.begin(), vec.end());                               \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            auto vec = get_constant_value<int, int64_t>(node.input()[index]); \
            dst.assign(vec.begin(), vec.end());                               \
        }                                                                     \
    }

    // starts/stops
    axis_t starts, stops;
    bool use_opset_1 = node.input().size() == 1;
    if (use_opset_1)
    {
        // opset 1
        starts = get_attribute<axis_t>(node, "starts").value();
        stops = get_attribute<axis_t>(node, "ends").value();
    }
    else
    {
        // opset 10/11/13
        GET_ATTRIBUTE(1, starts)

        GET_ATTRIBUTE(2, stops)
    }
    assert(starts.size() == stops.size());
    assert(starts.size() <= ndim);
    assert(stops.size() <= ndim);

    // optional axes
    axis_t axes;
    if (use_opset_1)
    {
        const auto axes_attr = get_attribute<axis_t>(node, "axes");
        if (axes_attr)
            axes = axes_attr.value();
    }
    else if (node.input().size() >= 4)
    {
        GET_ATTRIBUTE(3, axes)
    }

    if (axes.empty())
    {
        axes.assign(ndim, 0);
        std::iota(axes.begin(), axes.end(), 0);
    }

    // optional steps
    axis_t steps;
    if (node.input().size() >= 5)
    {
        GET_ATTRIBUTE(4, steps);
        assert(steps.size() == axes.size());
    }

    // fill begins/ends/strides
    axis_t begins(ndim, 0);
    axis_t ends(input_shape.begin(), input_shape.end());
    axis_t strides(ndim, 1);
    for (auto i = 0; i < ndim; ++i)
    {
        const auto it = std::find_if(axes.begin(), axes.end(),
            [i, ndim](const auto axis) { return real_axis(axis, ndim) == i; });
        if (it != axes.end())
        {
            auto idx = std::distance(axes.begin(), it);
            auto max = static_cast<int>(input_shape[i]);
            auto min = (-1) * max - 1;

            // check starts
            begins[i] = starts[idx] < min ? min : starts[idx] > max ? max
                                                                    : starts[idx];

            // check stops
            ends[i] = stops[idx] < min ? min : stops[idx] > max ? max
                                                                : stops[idx];

            // check steps
            if (!steps.empty())
            {
                assert(steps[idx] != 0);
                strides[i] = steps[idx];
            }

            // fixup begins
            if ((strides[i] > 0 && ends[i] > begins[i]) || (strides[i] < 0 && ends[i] < begins[i]))
            {
                begins[i] = begins[i] == min ? min + 1 : begins[i];
                begins[i] = begins[i] == max ? max - 1 : begins[i];
            }
            if (begins[i] < 0)
                begins[i] += max;
            if (ends[i] < 0)
                ends[i] += max;
        }
    }

    auto sl = graph_.emplace<slice>(input_type, input_shape, begins, ends, strides, 0, 0, 0, 0, 0);
    sl->name(generate_name(node) + "(Slice)");

    input_tensors_.emplace(&sl->input(), input);
    output_tensors_.emplace(node.output()[0], &sl->output());
}
