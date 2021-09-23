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
#include "tflite_importer.h"
#include <nncase/importer/util.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace flatbuffers;

tflite_importer::tflite_importer(std::span<const uint8_t> model, graph &graph)
    : model_(tflite::GetModel(model.data())), subgraph_(model_->subgraphs()->Get(0)), graph_(graph)
{
    flatbuffers::Verifier verifier(model.data(), model.size());
    if (!tflite::VerifyModelBuffer(verifier))
        throw std::runtime_error("Invalid tflite model");
}

void tflite_importer::import(const import_options &options, std::string &real_inlayout, std::string &real_outlayout)
{
    auto &operators = *subgraph_->operators();
    for (auto &&op : operators)
    {
        convert_op(*op);
    }

    std::unordered_map<int32_t, output_connector *> created_inputs;
    std::unordered_map<int32_t, input_connector *> created_outputs;

    // create inputs
    for (auto &&in : *subgraph_->inputs())
    {
        auto &tensor = *subgraph_->tensors()->Get(in);
        auto shape = get_shape(tensor.shape());
        auto type = to_data_type(tensor.type());

        auto node = graph_.emplace<input_node>(type, shape);
        node->name(tensor.name()->string_view());
        created_inputs.emplace(in, &node->output());
        real_inlayout = "NHWC";
    }

    std::vector<int32_t> outputs;
    if (options.output_arrays.empty())
    {
        for (auto &&out : *subgraph_->outputs())
        {
            outputs.emplace_back(out);
        }
    }
    else
    {
        for (auto &&name : options.output_arrays)
        {
            bool found = false;
            size_t i = 0;
            for (auto &&t : *subgraph_->tensors())
            {
                auto t_name = t->name();
                if (t_name && t_name->string_view() == name)
                {
                    outputs.emplace_back(i);
                    found = true;
                    break;
                }

                i++;
            }

            if (!found)
            {
                throw std::runtime_error("Cannot find output tensor: " + name);
            }
        }
    }

    // create outputs
    for (auto &&out : outputs)
    {
        auto &tensor = *subgraph_->tensors()->Get(out);
        auto shape = get_shape(tensor.shape());
        auto type = to_data_type(tensor.type());
        auto node = graph_.emplace<output_node>(type, shape);
        node->name(tensor.name()->string_view());
        created_outputs.emplace(out, &node->input());
        real_outlayout = "NHWC";
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
                auto con = graph_.emplace<constant>(type, shape, std::as_bytes(std::span(data->data(), data->data() + data->size())));
                con->name(tensor.name()->string_view());
                link_output_tensor(in.second, &con->output());
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
        auto in = created_outputs.find(out.first);
        if (in != created_outputs.end())
        {
            in->second->connect(*out.second);
        }
    }

    // outputs that connect to inputs or constants
    for (auto &&out : created_outputs)
    {
        if (!out.second->connection())
        {
            auto &tensor = *subgraph_->tensors()->Get(out.first);
            auto &buffer = *model_->buffers()->Get(tensor.buffer());
            auto data = buffer.data();

            if (data)
            {
                auto type = to_data_type(tensor.type());
                auto shape = get_shape(tensor.shape());
                auto con = graph_.emplace<constant>(type, shape, std::as_bytes(std::span(data->data(), data->data() + data->size())));
                con->name(tensor.name()->str() + "/const");
                out.second->connect(con->output());
            }
            else
            {
                auto in = created_inputs.find(out.first);
                if (in != created_inputs.end())
                    out.second->connect(*in->second);
            }
        }
    }

    graph_.dce();
}

void tflite_importer::convert_op(const tflite::Operator &op)
{
    auto opcode = model_->operator_codes()->Get(op.opcode_index());
    // Compatible with older version model
    auto builtin_code = static_cast<tflite::BuiltinOperator>(std::max(static_cast<int32_t>(opcode->deprecated_builtin_code()), static_cast<int32_t>(opcode->builtin_code())));
#define DEFINE_OPCODE(opcode)                             \
    if (builtin_code == tflite::BuiltinOperator_##opcode) \
        return convert_op_##opcode(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error(std::string("Unsupported tflite opcode: ") + tflite::EnumNameBuiltinOperator(builtin_code));
}

quant_param_t tflite_importer::to_quant_param(const tflite::QuantizationParameters *param)
{
    // TODO: consider of by axis quant
    return { (int32_t)param->zero_point()->Get(0), param->scale()->Get(0) };
}

void tflite_importer::add_convert(ir::input_connector &next_input, const tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type)
{
    auto ct = nncase::importer::add_prev_node<ir::convert>(graph_, next_input, to_data_type(tensor.type()), get_shape(tensor.shape()), to_type);
    link_input_tensor(&ct->input(), tf_id);
}

void tflite_importer::input_convert_to_type(ir::input_connector &next_input, const tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type)
{
    auto input_type = to_data_type(tensor.type());
    if (input_type != to_type)
    {
        add_convert(next_input, tensor, tf_id, to_type);
    }
    else
    {
        link_input_tensor(&next_input, tf_id);
    }
}