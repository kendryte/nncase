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
#include "caffe_importer.h"
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <nncase/importer/importer.h>
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
//using namespace nncase::runtime;
using namespace caffe;
using namespace std::string_view_literals;

caffe_importer::caffe_importer(std::span<const uint8_t> model, std::span<const uint8_t> prototxt, ir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid Caffe model");

    caffe::NetParameter proto;
    google::protobuf::io::ArrayInputStream input_stream(prototxt.data(), static_cast<int>(prototxt.size_bytes()));
    bool success = google::protobuf::TextFormat::Parse(&input_stream, &proto);
    if (!success)
    {
        throw std::runtime_error("read prototxt failed");
    }
    prototxt_ = proto;
}

void caffe_importer::import(std::string &real_layout)
{
    // TODO: support change input node layout
    real_layout = "NCHW";

    for (int i = 0; i < prototxt_.layer_size(); i++)
        convert_op(prototxt_.layer(i), model_);

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
            assert(!"Cannot find associated output node");
        }
    }

    // outputs
    std::unordered_set<std::string> used_inputs;
    std::unordered_set<std::string> seen_outputs;
    for (int i = 0; i < prototxt_.layer_size(); i++)
    {
        auto &layer = prototxt_.layer(i);
        for (auto &b : layer.bottom())
            used_inputs.emplace(b);
    }

    for (int i = 0; i < prototxt_.layer_size(); i++)
    {
        auto &layer = prototxt_.layer(i);
        for (auto &t : layer.top())
        {
            if (!used_inputs.contains(t)
                && !seen_outputs.contains(t))
            {
                seen_outputs.emplace(t);

                auto out = output_tensors_.at(t);
                auto node = graph_.emplace<output_node>(out->type(), out->shape());
                node->name(t);
                out->connect(node->input());
            }
        }
    }
}

void caffe_importer::convert_op(const LayerParameter &op, caffe::NetParameter caffemodel)
{
    auto type = op.type();

#define DEFINE_OPCODE(opcode) \
    if (type == #opcode##sv)  \
        return convert_op_##opcode(op, caffemodel);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported Caffe opcode: " + type);
}
