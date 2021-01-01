/* Copyright 2020 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/concat.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CONCATENATION)
{
    std::vector<shape_t> inputs_shape;
    auto &options = *op.builtin_options_as_ConcatenationOptions();

    for (auto &&in : *op.inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        inputs_shape.emplace_back(get_shape(tensor.shape()));
    }

    auto &type_tensor = get_tensor(op.inputs(), 0);
    datatype_t dtype = to_data_type(type_tensor.type());
    auto con = graph_.emplace<concat>(dtype, inputs_shape, options.axis());
    con->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/concat");

    for (size_t i = 0; i < op.inputs()->size(); i++)
        link_input_tensor(&con->input_at(i), op.inputs()->Get(flatbuffers::uoffset_t(i)));

    link_output_tensor(op.outputs()->Get(0), &con->output());
}

DEFINE_TFLITE_LOWER(PACK)
{
    std::vector<shape_t> inputs_shape;

    // TODO: Add reshapes
    for (auto &&in : *op.inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        auto shape = get_shape(tensor.shape());
        shape.insert(shape.begin(), 1);
        inputs_shape.emplace_back(shape);
    }

    auto &type_tensor = get_tensor(op.inputs(), 0);
    datatype_t dtype = to_data_type(type_tensor.type());
    auto con = graph_.emplace<concat>(dtype, inputs_shape, 0);
    con->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/concat");

    for (size_t i = 0; i < op.inputs()->size(); i++)
        link_input_tensor(&con->input_at(i), op.inputs()->Get(flatbuffers::uoffset_t(i)));

    link_output_tensor(op.outputs()->Get(0), &con->output());
}
