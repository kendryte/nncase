/* Copyright 2019 Canaan Inc.
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
#include <importer/importer.h>
#include <ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
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
            auto out_it = created_inputs.find(in.second);
            if (out_it != created_inputs.end())
            {
                in.first->connect(*out_it->second);
            }
            else
            {
                // image
                if (in.first->shape().size() == 4)
                {
                    auto node = graph_.emplace<input_node>(in.first->type(), nhwc_to_nchw(in.first->shape()));
                    auto sur_trans = nchw_to_nhwc(node->output().type(), node->output().shape());
                    sur_trans->input().connect(node->output());
                    in.first->connect(sur_trans->output());
                    created_inputs.emplace(in.second, &node->output());
                }
                else
                {
                    auto node = graph_.emplace<input_node>(in.first->type(), in.first->shape());
                    in.first->connect(node->output());
                    created_inputs.emplace(in.second, &node->output());
                }
            }
        }
    }

    // outputs
    for (auto &&out : output_tensors_)
    {
        if (out.second->connections().empty())
        {
            // image
            if (out.second->shape().size() == 4)
            {
                auto pre_trans = nhwc_to_nchw(out.second->type(), out.second->shape());
                auto node = graph_.emplace<output_node>(pre_trans->output().type(), pre_trans->output().shape());
                pre_trans->input().connect(*out.second);
                node->input().connect(pre_trans->output());
            }
            else
            {
                auto node = graph_.emplace<output_node>(out.second->type(), out.second->shape());
                out.second->connect(node->input());
            }
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
