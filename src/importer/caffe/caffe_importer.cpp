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
#include "caffe_importer.h"
#include <fstream>
#include <nncase/importer/importer.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/graph.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
//using namespace nncase::runtime;
using namespace caffe;
using namespace std::string_view_literals;

static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        throw std::runtime_error("open prototxt failed.");
    }
    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    std::string p;
    google::protobuf::TextFormat::PrintToString(*message, &p);

    fs.close();

    return success;
}

caffe_importer::caffe_importer(std::span<const uint8_t> model, const char *prototxt, ir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid Caffe model");

    caffe::NetParameter proto;
    bool s0 = read_proto_from_text(prototxt, &proto);
    if (!s0)
    {
        throw std::runtime_error("read prototxt failed");
    }
    prototxt_ = proto;
}

void caffe_importer::import([[maybe_unused]]const struct import_options &options)
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

void caffe_importer::convert_op(const LayerParameter &op, [[maybe_unused]]caffe::NetParameter caffemodel)
{
    auto type = op.type();

#define DEFINE_OPCODE(opcode) \
    if (type == #opcode##sv)  \
        return convert_op_##opcode(op, caffemodel);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported Caffe opcode: " + type);
}

// graph nncase::importer::import_caffe(xtl::span<const uint8_t> model)
// {
//     graph graph;
//     caffe_importer(model, graph).import();
//     return graph;
// }
