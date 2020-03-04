/* Copyright 2019-2020 Canaan Inc.
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
#include "tflite_importer.h"
#include <hlir/ops/constant.h>
#include <importer/importer.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::hlir;
using namespace flatbuffers;

tflite_importer::tflite_importer(xtl::span<const uint8_t> model, graph &graph)
    : model_(tflite::GetModel(model.data())), subgraph_(model_->subgraphs()->Get(0)), graph_(graph)
{
    flatbuffers::Verifier verifier(model.data(), model.size());
    if (!tflite::VerifyModelBuffer(verifier))
        throw std::runtime_error("Invalid tflite model");
}

void tflite_importer::import()
{
    auto &operators = *subgraph_->operators();
    for (auto &&op : operators)
        convert_op(*op);

    std::unordered_map<int32_t, output_connector *> created_inputs;
    std::unordered_map<int32_t, input_connector *> created_outputs;

    // create inputs
    for (auto &&in : *subgraph_->inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        auto shape = get_shape(tensor.shape());
        auto type = to_data_type(tensor.type());
        // image
        if (shape.size() == 4)
        {
            auto node = graph_.emplace<input_node>(type, nhwc_to_nchw(shape));
            node->name(tensor.name()->string_view());
            auto sur_trans = nchw_to_nhwc(node->output().type(), node->output().shape());
            sur_trans->input().connect(node->output());
            created_inputs.emplace(in, &sur_trans->output());
        }
        else
        {
            auto node = graph_.emplace<input_node>(type, shape);
            node->name(tensor.name()->string_view());
            created_inputs.emplace(in, &node->output());
        }
    }

    // create outputs
    for (auto &&out : *subgraph_->outputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(out);
        auto shape = get_shape(tensor.shape());
        auto type = to_data_type(tensor.type());
        // image
        if (shape.size() == 4)
        {
            auto pre_trans = nhwc_to_nchw(type, shape);
            auto node = graph_.emplace<output_node>(pre_trans->output().type(), pre_trans->output().shape());
            node->name(tensor.name()->string_view());
            node->input().connect(pre_trans->output());
            created_outputs.emplace(out, &pre_trans->input());
        }
        else
        {
            auto node = graph_.emplace<output_node>(type, shape);
            node->name(tensor.name()->string_view());
            created_outputs.emplace(out, &node->input());
        }
    }

    // connect tensors
    for (auto &&in : input_tensors_)
    {
        auto out_it = output_tensors_.find(in.second);
        if (out_it != output_tensors_.end())
        {
            in.first->connect(*out_it->second);
        }
        else
        {
            auto &tensor = *subgraph_->tensors()->Get(in.second);
            auto &buffer = *model_->buffers()->Get(tensor.buffer());
            auto data = buffer.data();

            if (data)
            {
                auto type = to_data_type(tensor.type());
                auto shape = get_shape(tensor.shape());
                auto con = graph_.emplace<constant>(type, shape, std::vector<uint8_t>(data->begin(), data->end()));
                output_tensors_.emplace(in.second, &con->output());
                in.first->connect(con->output());
            }
        }
    }

    // inputs
    for (auto &&in : input_tensors_)
    {
        if (!in.first->connection())
        {
            auto out = created_inputs.at(in.second);
            in.first->connect(*out);
        }
    }

    // outputs
    for (auto &&out : output_tensors_)
    {
        if (out.second->connections().empty())
        {
            auto in = created_outputs.at(out.first);
            in->connect(*out.second);
        }
    }
}

void tflite_importer::convert_op(const tflite::Operator &op)
{
    auto opcode = model_->operator_codes()->Get(op.opcode_index());
    auto builtin_code = opcode->builtin_code();

#define DEFINE_OPCODE(opcode)                             \
    if (builtin_code == tflite::BuiltinOperator_##opcode) \
        return convert_op_##opcode(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error(std::string("Not supported tflite opcode: ") + tflite::EnumNameBuiltinOperator(builtin_code));
}

graph nncase::importer::import_tflite(xtl::span<const uint8_t> model)
{
    graph graph;
    tflite_importer(model, graph).import();
    return graph;
}
