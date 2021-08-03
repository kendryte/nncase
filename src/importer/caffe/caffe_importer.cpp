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
    std::cout<<"test protobuf1"<<std::endl;
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::cout<<"test protobuf2"<<std::endl;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid Caffe model");

    caffe::NetParameter proto;
    google::protobuf::io::ArrayInputStream input_stream(prototxt.data(), static_cast<int>(prototxt.size_bytes()));
    std::cout<<"test protobuf3"<<std::endl;
    bool success = google::protobuf::TextFormat::Parse(&input_stream, &proto);
    std::cout<<"test protobuf4"<<std::endl;
    if (!success)
    {
        throw std::runtime_error("read prototxt failed");
    }
    prototxt_ = proto;
}

void caffe_importer::import()
{
    for (int i = 0; i < prototxt_.layer_size(); i++)
        convert_op(prototxt_.layer(i), model_);

    std::unordered_map<std::string_view, output_connector *> created_inputs;

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
    for (auto &&out : output_tensors_)
    {
        if (out.second->connections().empty())
        {
            auto node = graph_.emplace<output_node>(out.second->type(), out.second->shape());
            node->name(out.first);
            out.second->connect(node->input());
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
