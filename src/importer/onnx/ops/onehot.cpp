/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/onehot.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/importer/util.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace onnx;

template<class Node, class... Args>
Node* add_node(ir::graph &graph, input_connector &next_input, Args&&... args)
{
    auto node = graph.emplace<Node>(std::forward<Args>(args)...);
    next_input.connect(node->output());
    return node;
}

void onnx_importer::convert_op_OneHot(const NodeProto &node)
{
    const auto &indices = node.input()[0];
    const auto &depth = node.input()[1];
    const auto &values = node.input()[2];
    const auto &output = node.output()[0];
    const auto type = get_datatype(output).value();

    auto indices_shape = get_shape(indices);
    auto out_shape = get_shape(output);

    int32_t axis = 0;
    const auto axis_attr = get_attribute<int32_t>(node, "axis");
    axis = get_positive<int32_t>(axis_attr.value(), out_shape.size());

    auto oh = graph_.emplace<onehot>(type, indices_shape, out_shape, axis);
    const auto &op_name { generate_name(node) };
    oh->name(op_name + "(OneHot)");

    const auto indices_type = get_datatype(indices).value();
    if (indices_type != dt_int32)
    {
        auto ct = add_node<convert>(graph_, oh->indices(), indices_type, indices_shape, dt_int32);
        link_input_tensor(&ct->input(), indices);
    }
    else
    {
        link_input_tensor(&oh->indices(), indices);
    }
    const auto depth_type = get_datatype(depth).value();
    if(depth_type != dt_int32)
    {
        auto ct = add_node<convert>(graph_, oh->depth(), depth_type, get_shape(depth), dt_int32);
        link_input_tensor(&ct->input(), depth);
    }
    else
    {
        link_input_tensor(&oh->depth(), depth);
    }

    axis_t values_begin = {0};
    axis_t off_values_end = {1};
    axis_t values_end = {2};

    auto value_type = get_datatype(values).value();
    auto value_shape = get_shape(values);
    auto sl_off = add_node<slice>(graph_, oh->off_value(), value_type, value_shape, values_begin, off_values_end);
    link_input_tensor(&sl_off->input(), values);
    auto sl_on = add_node<slice>(graph_, oh->on_value(), value_type, value_shape, off_values_end, values_end);
    link_input_tensor(&sl_on->input(), values);
    link_output_tensor(output, &oh->output());
}