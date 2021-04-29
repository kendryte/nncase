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

#include <hlir/ops/split.h>
#include <hlir/graph.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Split(const NodeProto& node)
{
    // Input node
    const auto &input { node.input()[0] };
    const auto input_shape { get_shape(input) };
    const auto data_type { get_datatype(input).value() };

    // Axis to split on
    const auto &axis { get_attribute<int64_t>(node, "axis") };
    const auto &split_attr = node.attribute(1); // 0 is axis, 1 is split

    if(!axis) throw runtime_error("No axis specified");
    if(split_attr.name() != "split") throw runtime_error("Unsupported split type; no split attribute provided");

    auto splits = vector<int64_t>();

    // Split could be either a:
    // 1D tensor which is a list of the size of each split element
    // or an int, defining number of splits
    if(split_attr.has_i()) {
        // Is an integer split
        auto num_splits = split_attr.i();
        // Break input axis into even splits
        auto axis_dimension = input_shape[*axis];
        if(axis_dimension % num_splits != 0) throw runtime_error("Invalid split size.");
        int64_t split_size = axis_dimension / num_splits;
        for(int i = 0; i < split_size; i ++) {
            splits.push_back(split_size);
        }
    } else if(split_attr.ints_size() > 0){
        // Is a 1D tensor
        // Copy to our splits vector
        for(const auto &e : split_attr.ints()) {
            splits.push_back(e);
        }
    }

    // Validate our num splits matches our output size
    if(node.output_size() != splits.size()) throw runtime_error("Invalid split; output layers to do not match split dimensions");

    // Add to our graph
    auto graph_node {graph_.emplace<split>(data_type, input_shape, *axis, splits) };
    // Add our input tensor
    input_tensors_.emplace(&graph_node->input(), input);
    // Add all of our output tensors
    for(int i = 0; i < graph_node->outputs().size(); i ++) {
        output_tensors_.emplace(node.output(i), &graph_node->output_at(i));
    }
}
