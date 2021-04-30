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

#include <hlir/ops/upsample.h>
#include <hlir/graph.h>

using namespace std;

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;

using namespace onnx;

void onnx_importer::convert_op_Upsample(const NodeProto& node)
{
    // Main tensor_input node
    const auto &tensor_input {node.input(0) };
    shape_t tensor_shape {get_shape(tensor_input) };
    const auto data_type { get_datatype(tensor_input).value() };

    // Scales tensor_input
    const auto &scales_input {node.input(1)};
    const auto &scales_data {get_initializer(scales_input)};

    // Assert we have a valid initializer for our scales_data
    assert(scales_data);

    vector<float> scales;
    for(const auto &scale : scales_data->float_data()) {
        scales.emplace_back(scale);
    }

    // Get scaling mode
    const auto mode {get_attribute<string>(node, "mode")};
    // Only supported mode
    if(mode != "nearest") {
        throw runtime_error("Only nearest mode supported for Upsample.");
    }

    auto graph_node {graph_.emplace<upsample>(data_type, tensor_shape, scales)};
    // Add the 2 inputs and 1 output
    input_tensors_.emplace(&graph_node->input_at(0), tensor_input);
    output_tensors_.emplace(node.output(0), &graph_node->output_at(0));
}
