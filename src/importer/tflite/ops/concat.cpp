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
#include "../tflite_importer.h"
#include <nncase/ir/ops/bitcast.h>
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
    std::vector<bitcast *> inputs_rshape;
    auto &options = *op.builtin_options_as_PackOptions();

    for (auto &&in : *op.inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        auto old_shape = get_shape(tensor.shape());
        auto new_shape = old_shape;
        new_shape.insert(new_shape.begin() + options.axis(), 1);
        auto rshape = graph_.emplace<bitcast>(to_data_type(tensor.type()), old_shape, new_shape);
        rshape->name(std::string(tensor.name()->string_view()) + "/reshape");
        link_input_tensor(&rshape->input(), in);
        inputs_shape.emplace_back(new_shape);
        inputs_rshape.emplace_back(rshape);
    }

    auto &type_tensor = get_tensor(op.inputs(), 0);
    datatype_t dtype = to_data_type(type_tensor.type());
    auto con = graph_.emplace<concat>(dtype, inputs_shape, options.axis());
    con->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/concat");
    auto rshape = graph_.emplace<bitcast>(dtype, con->output().shape(), get_shape(get_tensor(op.outputs(), 0).shape()));
    rshape->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/reshape");
    rshape->input().connect(con->output());

    for (size_t i = 0; i < inputs_rshape.size(); i++)
        con->input_at(i).connect(inputs_rshape[i]->output());
    link_output_tensor(op.outputs()->Get(0), &rshape->output());
}
