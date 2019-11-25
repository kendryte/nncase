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
#include "caffe_importer.h"
#include <fstream>
#include <importer/importer.h>
#include <ir/ops/constant.h>
#include <runtime/binary_reader.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace nncase::runtime;
using namespace caffe;
using namespace std::string_view_literals;

caffe_importer::caffe_importer(xtl::span<const uint8_t> model, ir::graph &graph)
    : graph_(graph)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid PaddlePaddle model");
}

void caffe_importer::import()
{
    for (int i = 0; i < model_.layer_size(); i++)
        convert_op(model_.layer(i));
}

void caffe_importer::convert_op(const LayerParameter &op)
{
    auto type = op.type();

#define DEFINE_OPCODE(opcode) \
    if (type == #opcode##sv)  \
        return convert_op_##opcode(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported Caffe opcode: " + type);
}

graph nncase::importer::import_caffe(xtl::span<const uint8_t> model)
{
    graph graph;
    caffe_importer(model, graph).import();
    return graph;
}
